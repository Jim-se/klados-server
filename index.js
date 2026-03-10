const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const { OpenRouter } = require('@openrouter/sdk');
const { OpenAI } = require('openai');
const { GoogleGenAI } = require('@google/genai');
const axios = require('axios');
const { PassThrough } = require('stream');

dotenv.config();

const app = express();
const port = process.env.PORT || 3001;

const xss = require('xss');
const helmet = require('helmet');

// Middlewares - Strict CORS Protection
const allowedOrigins = [
    'http://localhost:5173',
    'http://localhost:5174',
    'http://localhost:3000'
];

if (process.env.FRONTEND_URL) {
    allowedOrigins.push(process.env.FRONTEND_URL);
}

app.use(cors({
    origin: function (origin, callback) {
        // Allow requests with no origin (like Postman or server-to-server)
        if (!origin || allowedOrigins.includes(origin)) {
            callback(null, true);
        } else {
            console.warn(`[CORS] Rejected request from unauthorized origin: ${origin}`);
            callback(new Error('Blocked by CORS policy: Origin not allowed'));
        }
    },
    methods: ['GET', 'POST', 'PATCH', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization'],
    credentials: true, // Allow authorization headers to be sent
    maxAge: 86400 // Cache preflight options for 24 hours
}));

// Content Security Policy (API constraints)
app.use(helmet({
    contentSecurityPolicy: {
        directives: {
            defaultSrc: ["'none'"],
            scriptSrc: ["'none'"],
        }
    }
}));

// Recursive input sanitization to filter XSS on arrival
const sanitizeBody = (obj) => {
    if (typeof obj === 'string') {
        return xss(obj);
    }
    if (Array.isArray(obj)) {
        return obj.map(sanitizeBody);
    }
    if (typeof obj === 'object' && obj !== null) {
        const sanitized = {};
        for (const [key, value] of Object.entries(obj)) {
            sanitized[key] = sanitizeBody(value);
        }
        return sanitized;
    }
    return obj;
};

const sanitizeParsedBody = (req, res, next) => {
    if (req.body) {
        req.body = sanitizeBody(req.body);
    }
    next();
};

const createContentLengthGuard = (maxBytes) => (req, res, next) => {
    const headerValue = req.headers['content-length'];
    const contentLength = Number(headerValue);

    if (Number.isFinite(contentLength) && contentLength > maxBytes) {
        return res.status(413).json({ error: 'Payload too large' });
    }

    next();
};

const createJsonBodyMiddleware = ({ limit, maxBytes }) => ([
    createContentLengthGuard(maxBytes),
    express.json({ limit }),
    sanitizeParsedBody,
]);

// --- Middleware to verify User Auth & Create Authenticated Client ---
const { createClient } = require('@supabase/supabase-js');
const authenticateUser = async (req, res, next) => {
    const token = req.headers.authorization?.split(' ')[1];

    if (!token) {
        return res.status(401).json({ error: 'Unauthorized: No token provided' });
    }

    try {
        const userClient = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY, {
            global: { headers: { Authorization: `Bearer ${token}` } }
        });

        const { data: { user }, error } = await userClient.auth.getUser();

        if (error || !user) {
            return res.status(401).json({ error: 'Unauthorized: Invalid token' });
        }

        req.userClient = userClient;
        req.user = user;
        next();
    } catch (err) {
        console.error('[AUTH ERROR]:', err.message);
        res.status(500).json({ error: 'Internal Auth Error' });
    }
};

// --- LLM Security Middlewares ---
const rateLimit = require('express-rate-limit');
const requestRateLimitKey = (req) => req.user ? req.user.id : req.ip;

const llmRateLimiter = rateLimit({
    windowMs: 60 * 1000, // 1 minute
    max: 20, // 20 requests per minute
    keyGenerator: requestRateLimitKey, // Limit per properly tracked user instead of shared IP
    message: { error: 'Too many requests, please try again later.' }
});

const dbWriteRateLimiter = rateLimit({
    windowMs: 60 * 1000,
    max: 60,
    keyGenerator: requestRateLimitKey,
    message: { error: 'Too many write requests, please try again later.' }
});

const maxPayloadSize = (req, res, next) => {
    // 5MB limit roughly (base64 images can be big, but prevent massive memory exhaustion)
    if (req.body && JSON.stringify(req.body).length > 5000000) {
        console.warn(`[SECURITY] Blocked oversized payload from User: ${req.user?.id || 'Unknown'}`);
        return res.status(413).json({ error: 'Payload too large' });
    }
    next();
};

const llmSecurity = [authenticateUser, llmRateLimiter];
const dbWriteSecurity = [authenticateUser, dbWriteRateLimiter];

const JSON_LIMITS = {
    llm: { limit: '5mb', maxBytes: 5 * 1024 * 1024 },
    message: { limit: '2mb', maxBytes: 2 * 1024 * 1024 },
    bugReport: { limit: '256kb', maxBytes: 256 * 1024 },
    small: { limit: '64kb', maxBytes: 64 * 1024 },
};

const llmJsonBody = createJsonBodyMiddleware(JSON_LIMITS.llm);
const messageJsonBody = createJsonBodyMiddleware(JSON_LIMITS.message);
const bugReportJsonBody = createJsonBodyMiddleware(JSON_LIMITS.bugReport);
const smallJsonBody = createJsonBodyMiddleware(JSON_LIMITS.small);

const parseModelAllowlist = (envValue) => {
    if (!envValue) {
        return null;
    }

    const values = envValue
        .split(',')
        .map((value) => value.trim())
        .filter(Boolean);

    return values.length > 0 ? new Set(values) : null;
};

const OPENROUTER_ALLOWED_MODELS = parseModelAllowlist(process.env.OPENROUTER_ALLOWED_MODELS);
const OPENAI_ALLOWED_MODELS = parseModelAllowlist(process.env.OPENAI_ALLOWED_MODELS);
const GEMINI_ALLOWED_MODELS = parseModelAllowlist(process.env.GEMINI_ALLOWED_MODELS);
const TRACKED_ALLOWED_MODELS = parseModelAllowlist(process.env.TRACKED_ALLOWED_MODELS);

const normalizeModelId = (value) => typeof value === 'string' ? value.trim() : '';

const createModelAccessGuard = ({ allowedModels, bodyKey = 'model', defaultModel = '' }) => (
    (req, res, next) => {
        const requestedModel = normalizeModelId(req.body?.[bodyKey] || defaultModel);

        if (!requestedModel) {
            return res.status(400).json({ error: 'A model identifier is required.' });
        }

        if (allowedModels && !allowedModels.has(requestedModel)) {
            return res.status(403).json({ error: 'Requested model is not enabled for this server.' });
        }

        req.allowedModel = requestedModel;
        next();
    }
);

// --- Service Clients ---

const openRouterClient = new OpenRouter({
    apiKey: process.env.OPENROUTER_API_KEY,
});

const openaiClient = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

const genAI = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

const USER_ACCOUNTING_COLUMNS = 'id, email, full_name, tier, billing_period_start, billing_period_end, total_requests, lifetime_cost, current_period_cost, created_at';
const PROVIDER_ALIASES = {
    'arcee-ai': 'arcee',
};

const toFiniteNumber = (value) => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : 0;
};

const toTokenCount = (value) => {
    const parsed = Math.round(toFiniteNumber(value));
    return parsed > 0 ? parsed : 0;
};

const roundCost = (value) => Number(toFiniteNumber(value).toFixed(8));

const addOneMonth = (dateLike) => {
    const baseDate = dateLike ? new Date(dateLike) : new Date();
    const safeDate = Number.isNaN(baseDate.getTime()) ? new Date() : baseDate;
    const nextDate = new Date(safeDate);
    nextDate.setMonth(nextDate.getMonth() + 1);
    return nextDate;
};

const deriveProviderFromModel = (model) => {
    if (typeof model !== 'string' || !model) {
        return 'unknown';
    }

    const [prefix] = model.split('/');
    return PROVIDER_ALIASES[prefix] || prefix || 'unknown';
};

const buildBillingWindow = (createdAt) => {
    const start = createdAt ? new Date(createdAt) : new Date();
    const safeStart = Number.isNaN(start.getTime()) ? new Date() : start;
    return {
        start: safeStart,
        end: addOneMonth(safeStart),
    };
};

const advanceBillingWindow = (startLike, endLike, now = new Date()) => {
    let start = startLike ? new Date(startLike) : new Date(now);
    let end = endLike ? new Date(endLike) : addOneMonth(start);

    if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime()) || end <= start) {
        return {
            start: new Date(now),
            end: addOneMonth(now),
            reset: true,
        };
    }

    let reset = false;
    while (end <= now) {
        start = new Date(end);
        end = addOneMonth(start);
        reset = true;
    }

    return { start, end, reset };
};

const ensureUserAccountingRow = async (userClient, user) => {
    const { data: existingUser, error } = await userClient
        .from('users')
        .select(USER_ACCOUNTING_COLUMNS)
        .eq('id', user.id)
        .maybeSingle();

    if (error && error.code !== 'PGRST116') {
        throw error;
    }

    const defaultWindow = buildBillingWindow(existingUser?.created_at || user.created_at);

    if (!existingUser) {
        const { data: insertedUser, error: insertError } = await userClient
            .from('users')
            .upsert({
                id: user.id,
                email: user.email ?? null,
                full_name: user.user_metadata?.full_name ?? null,
                tier: 'FREE',
                billing_period_start: defaultWindow.start.toISOString(),
                billing_period_end: defaultWindow.end.toISOString(),
                total_requests: 0,
                lifetime_cost: 0,
                current_period_cost: 0,
            })
            .select(USER_ACCOUNTING_COLUMNS)
            .single();

        if (insertError) {
            throw insertError;
        }

        return insertedUser;
    }

    const updates = {};

    if (user.email && existingUser.email !== user.email) {
        updates.email = user.email;
    }

    if (!existingUser.full_name && user.user_metadata?.full_name) {
        updates.full_name = user.user_metadata.full_name;
    }

    if (!existingUser.tier) {
        updates.tier = 'FREE';
    }

    if (!existingUser.billing_period_start) {
        updates.billing_period_start = defaultWindow.start.toISOString();
    }

    if (!existingUser.billing_period_end) {
        updates.billing_period_end = defaultWindow.end.toISOString();
    }

    if (existingUser.total_requests == null) {
        updates.total_requests = 0;
    }

    if (existingUser.lifetime_cost == null) {
        updates.lifetime_cost = 0;
    }

    if (existingUser.current_period_cost == null) {
        updates.current_period_cost = 0;
    }

    if (Object.keys(updates).length === 0) {
        return existingUser;
    }

    const { data: updatedUser, error: updateError } = await userClient
        .from('users')
        .update(updates)
        .eq('id', user.id)
        .select(USER_ACCOUNTING_COLUMNS)
        .single();

    if (updateError) {
        throw updateError;
    }

    return updatedUser;
};

const resolvePricingForModel = async (userClient, model) => {
    const { data: priceRow, error } = await userClient
        .from('prices')
        .select('provider, input_price_per_million, output_price_per_million')
        .eq('model', model)
        .maybeSingle();

    if (error && error.code !== 'PGRST116') {
        throw error;
    }

    if (!priceRow) {
        console.warn(`[USAGE] No price row found for model "${model}". Defaulting cost to 0.`);
    }

    return {
        provider: priceRow?.provider || deriveProviderFromModel(model),
        inputPricePerMillion: toFiniteNumber(priceRow?.input_price_per_million),
        outputPricePerMillion: toFiniteNumber(priceRow?.output_price_per_million),
    };
};

const collectTextFragments = (value) => {
    if (!value) return [];

    if (typeof value === 'string') {
        return value ? [value] : [];
    }

    if (Array.isArray(value)) {
        return value.flatMap(collectTextFragments);
    }

    if (typeof value === 'object') {
        if (typeof value.text === 'string') {
            return value.text ? [value.text] : [];
        }

        if (typeof value.content === 'string') {
            return value.content ? [value.content] : [];
        }

        if (Array.isArray(value.content)) {
            return value.content.flatMap(collectTextFragments);
        }

        if (Array.isArray(value.parts)) {
            return value.parts.flatMap(collectTextFragments);
        }
    }

    return [];
};

const estimateTokensFromText = (text) => {
    if (typeof text !== 'string' || !text) return 0;
    // Conservative heuristic to avoid under-reserving.
    return Math.ceil(text.length / 4);
};

const estimateInputTokensFromMessages = (messages) => {
    if (!Array.isArray(messages)) return 0;
    const text = messages
        .flatMap((msg) => collectTextFragments(msg?.content))
        .join(' ');
    return estimateTokensFromText(text);
};

const extractUsageTokens = (payload) => {
    const usage = payload?.usage;
    if (!usage || typeof usage !== 'object') return null;

    const inputTokens = toTokenCount(
        usage.prompt_tokens ?? usage.input_tokens ?? usage.promptTokens ?? usage.inputTokens ?? usage.promptTokenCount
    );
    const outputTokens = toTokenCount(
        usage.completion_tokens ??
        usage.output_tokens ??
        usage.completionTokens ??
        usage.outputTokens ??
        usage.candidatesTokenCount
    );

    if (!inputTokens && !outputTokens) {
        return null;
    }

    return { inputTokens, outputTokens };
};

const reserveUsageEvent = async (userClient, { provider, model, reservedCost, reservedInputTokens, reservedOutputTokens }) => {
    const { data, error } = await userClient
        .rpc('usage_reserve_request', {
            p_provider: provider,
            p_model: model,
            p_reserved_cost: reservedCost,
            p_reserved_input_tokens: reservedInputTokens ?? null,
            p_reserved_output_tokens: reservedOutputTokens ?? null,
        })
        .single();

    if (error) {
        throw error;
    }

    return data;
};

const finalizeUsageEvent = async (userClient, { usageEventId, cost, inputTokens, outputTokens }) => {
    const { error } = await userClient.rpc('usage_finalize_request', {
        p_usage_event_id: usageEventId,
        p_cost: cost,
        p_input_tokens: inputTokens ?? null,
        p_output_tokens: outputTokens ?? null,
    });

    if (error) {
        throw error;
    }
};

const cancelUsageEvent = async (userClient, { usageEventId, errorMessage }) => {
    const { error } = await userClient.rpc('usage_cancel_request', {
        p_usage_event_id: usageEventId,
        p_error: errorMessage ?? null,
    });

    if (error) {
        throw error;
    }
};

const recordUsageAggregate = async (userClient, { userId, model, provider, inputTokens, outputTokens, totalCost }) => {
    const usageDate = new Date().toISOString().slice(0, 10);
    const { data: existingUsage, error } = await userClient
        .from('usage')
        .select('id, total_input_tokens, total_output_tokens, total_cost')
        .eq('user_id', userId)
        .eq('model', model)
        .eq('provider', provider)
        .maybeSingle();

    if (error && error.code !== 'PGRST116') {
        throw error;
    }

    if (!existingUsage) {
        const { error: insertError } = await userClient.from('usage').insert({
            date: usageDate,
            user_id: userId,
            provider,
            model,
            total_input_tokens: inputTokens,
            total_output_tokens: outputTokens,
            total_cost: totalCost,
        });

        if (insertError) {
            throw insertError;
        }

        return;
    }

    const { error: updateError } = await userClient
        .from('usage')
        .update({
            total_input_tokens: toTokenCount(existingUsage.total_input_tokens) + inputTokens,
            total_output_tokens: toTokenCount(existingUsage.total_output_tokens) + outputTokens,
            total_cost: roundCost(toFiniteNumber(existingUsage.total_cost) + totalCost),
        })
        .eq('id', existingUsage.id);

    if (updateError) {
        throw updateError;
    }
};

const recordUserAggregate = async (userClient, user, totalCost) => {
    const currentUser = await ensureUserAccountingRow(userClient, user);
    const now = new Date();
    const { start, end, reset } = advanceBillingWindow(
        currentUser.billing_period_start || currentUser.created_at || user.created_at,
        currentUser.billing_period_end,
        now
    );

    const nextTotalRequests = toTokenCount(currentUser.total_requests) + 1;
    const nextLifetimeCost = roundCost(toFiniteNumber(currentUser.lifetime_cost) + totalCost);
    const currentPeriodBaseCost = reset ? 0 : toFiniteNumber(currentUser.current_period_cost);

    const { error } = await userClient
        .from('users')
        .update({
            tier: 'FREE',
            total_requests: nextTotalRequests,
            lifetime_cost: nextLifetimeCost,
            current_period_cost: roundCost(currentPeriodBaseCost + totalCost),
            billing_period_start: start.toISOString(),
            billing_period_end: end.toISOString(),
            updated_at: now.toISOString(),
        })
        .eq('id', user.id);

    if (error) {
        throw error;
    }
};

// --- Routes ---

/**
 * OpenRouter Proxy with Streaming Support
 */
app.post(
    '/api/openrouter/chat',
    ...llmSecurity,
    ...llmJsonBody,
    maxPayloadSize,
    createModelAccessGuard({ allowedModels: OPENROUTER_ALLOWED_MODELS }),
    async (req, res) => {
    let usageEventId = null;
    try {
        const { stream = false, model: _ignoredModel, ...requestBody } = req.body;
        const model = req.allowedModel;

        const pricing = await resolvePricingForModel(req.userClient, model);
        const inputTokenEstimate = estimateInputTokensFromMessages(requestBody.messages);
        const outputTokenBudget = toTokenCount(
            requestBody.max_tokens ?? requestBody.max_completion_tokens ?? requestBody.max_output_tokens ?? 1024
        );
        const reservedCost = roundCost(
            (inputTokenEstimate / 1_000_000) * pricing.inputPricePerMillion +
            (outputTokenBudget / 1_000_000) * pricing.outputPricePerMillion
        );

        const reservation = await reserveUsageEvent(req.userClient, {
            provider: pricing.provider,
            model,
            reservedCost,
            reservedInputTokens: inputTokenEstimate,
            reservedOutputTokens: outputTokenBudget,
        });

        if (!reservation?.allowed) {
            return res.status(402).json({
                error: 'Usage limit exceeded',
                reason: reservation?.reason ?? 'LIMIT',
                four_hour: {
                    spend: reservation?.four_hour_spend ?? null,
                    limit: reservation?.four_hour_limit ?? null,
                },
                month: {
                    spend: reservation?.monthly_spend ?? null,
                    limit: reservation?.monthly_limit ?? null,
                },
            });
        }

        usageEventId = reservation?.usage_event_id;

        const response = await axios.post('https://openrouter.ai/api/v1/chat/completions', {
            model,
            ...requestBody,
            stream,
        }, {
            headers: {
                'Authorization': `Bearer ${process.env.OPENROUTER_API_KEY}`,
                'Content-Type': 'application/json',
            },
            responseType: stream ? 'stream' : 'json'
        });

        if (stream) {
            res.setHeader('Content-Type', 'text/event-stream');
            res.setHeader('Cache-Control', 'no-cache');
            res.setHeader('Connection', 'keep-alive');

            const upstream = response.data;
            const tee = new PassThrough();
            let buffer = '';
            let latestUsage = null;
            let finalized = false;

            const finalizeOnce = async ({ inputTokens, outputTokens, cost, errorMessage }) => {
                if (!usageEventId || finalized) return;
                finalized = true;

                try {
                    if (errorMessage) {
                        await finalizeUsageEvent(req.userClient, {
                            usageEventId,
                            cost: reservedCost,
                            inputTokens: inputTokens ?? inputTokenEstimate,
                            outputTokens: outputTokens ?? outputTokenBudget,
                        });
                        return;
                    }

                    await finalizeUsageEvent(req.userClient, {
                        usageEventId,
                        cost,
                        inputTokens,
                        outputTokens,
                    });
                } catch (finalizeError) {
                    console.error('[USAGE] Failed to finalize usage_event:', finalizeError.message);
                }
            };

            tee.on('data', (chunk) => {
                buffer += chunk.toString('utf8');
                let newlineIndex;
                while ((newlineIndex = buffer.indexOf('\n')) !== -1) {
                    const line = buffer.slice(0, newlineIndex).trimEnd();
                    buffer = buffer.slice(newlineIndex + 1);
                    if (!line.startsWith('data:')) continue;

                    const payloadText = line.slice('data:'.length).trim();
                    if (!payloadText || payloadText === '[DONE]') continue;

                    try {
                        const payload = JSON.parse(payloadText);
                        const usage = extractUsageTokens(payload);
                        if (usage) {
                            latestUsage = usage;
                        }
                    } catch {
                        // ignore parse errors on non-JSON frames
                    }
                }
            });

            upstream.on('end', async () => {
                const usage = latestUsage;
                if (!usage) {
                    await finalizeOnce({ errorMessage: 'missing_usage' });
                    return;
                }

                const inputCost = roundCost((usage.inputTokens / 1_000_000) * pricing.inputPricePerMillion);
                const outputCost = roundCost((usage.outputTokens / 1_000_000) * pricing.outputPricePerMillion);
                const totalCost = roundCost(inputCost + outputCost);
                await finalizeOnce({
                    inputTokens: usage.inputTokens,
                    outputTokens: usage.outputTokens,
                    cost: totalCost,
                });
            });

            upstream.on('error', async (err) => {
                console.error('OpenRouter Stream Error:', err.message);
                await finalizeOnce({ errorMessage: err.message });
            });

            res.on('close', async () => {
                if (!upstream.destroyed) {
                    upstream.destroy();
                }
                await finalizeOnce({ errorMessage: 'client_disconnected' });
            });

            upstream.pipe(tee);
            tee.pipe(res);
        } else {
            const usage = extractUsageTokens(response.data);
            if (usageEventId) {
                try {
                    const inputCost = roundCost((toTokenCount(usage?.inputTokens) / 1_000_000) * pricing.inputPricePerMillion);
                    const outputCost = roundCost((toTokenCount(usage?.outputTokens) / 1_000_000) * pricing.outputPricePerMillion);
                    const totalCost = roundCost(inputCost + outputCost);

                    await finalizeUsageEvent(req.userClient, {
                        usageEventId,
                        cost: usage ? totalCost : reservedCost,
                        inputTokens: usage?.inputTokens ?? inputTokenEstimate,
                        outputTokens: usage?.outputTokens ?? outputTokenBudget,
                    });
                } catch (finalizeError) {
                    console.error('[USAGE] Failed to finalize usage_event:', finalizeError.message);
                }
            }
            res.json(response.data);
        }
    } catch (error) {
        if (usageEventId) {
            try {
                await cancelUsageEvent(req.userClient, { usageEventId, errorMessage: error.message });
            } catch (cancelError) {
                console.error('[USAGE] Failed to cancel usage_event:', cancelError.message);
            }
        }
        const upstreamStatus = error.response?.status || 500;
        const upstreamData = error.response?.data;

        console.error('OpenRouter Error:', upstreamData || error.message);

        if (upstreamData && typeof upstreamData === 'object' && !Buffer.isBuffer(upstreamData)) {
            return res.status(upstreamStatus).json(upstreamData);
        }

        return res.status(upstreamStatus).json({
            error: typeof upstreamData === 'string' && upstreamData
                ? upstreamData
                : error.message || 'Failed to fetch from OpenRouter',
        });
    }
});

/**
 * OpenAI Proxy
 */
app.post(
    '/api/openai/chat',
    ...llmSecurity,
    ...llmJsonBody,
    maxPayloadSize,
    createModelAccessGuard({ allowedModels: OPENAI_ALLOWED_MODELS }),
    async (req, res) => {
    let usageEventId = null;
    try {
        const { messages, max_tokens, max_completion_tokens } = req.body;
        const model = req.allowedModel;

        const pricing = await resolvePricingForModel(req.userClient, model);
        const inputTokenEstimate = estimateInputTokensFromMessages(messages);
        const outputTokenBudget = toTokenCount(max_tokens ?? max_completion_tokens ?? 1024);
        const reservedCost = roundCost(
            (inputTokenEstimate / 1_000_000) * pricing.inputPricePerMillion +
            (outputTokenBudget / 1_000_000) * pricing.outputPricePerMillion
        );

        const reservation = await reserveUsageEvent(req.userClient, {
            provider: pricing.provider,
            model,
            reservedCost,
            reservedInputTokens: inputTokenEstimate,
            reservedOutputTokens: outputTokenBudget,
        });

        if (!reservation?.allowed) {
            return res.status(402).json({ error: 'Usage limit exceeded', reason: reservation?.reason ?? 'LIMIT' });
        }

        usageEventId = reservation?.usage_event_id;

        const completion = await openaiClient.chat.completions.create({
            model,
            messages,
        });

        const usage = extractUsageTokens(completion);
        if (usageEventId) {
            const inputCost = roundCost((toTokenCount(usage?.inputTokens) / 1_000_000) * pricing.inputPricePerMillion);
            const outputCost = roundCost((toTokenCount(usage?.outputTokens) / 1_000_000) * pricing.outputPricePerMillion);
            const totalCost = roundCost(inputCost + outputCost);
            await finalizeUsageEvent(req.userClient, {
                usageEventId,
                cost: usage ? totalCost : reservedCost,
                inputTokens: usage?.inputTokens ?? inputTokenEstimate,
                outputTokens: usage?.outputTokens ?? outputTokenBudget,
            });
        }
        res.json(completion);
    } catch (error) {
        if (usageEventId) {
            try {
                await cancelUsageEvent(req.userClient, { usageEventId, errorMessage: error.message });
            } catch (cancelError) {
                console.error('[USAGE] Failed to cancel usage_event:', cancelError.message);
            }
        }
        console.error('OpenAI Error:', error.message);
        res.status(500).json({ error: 'Failed to fetch from OpenAI' });
    }
});

/**
 * Gemini Proxy
 */
app.post(
    '/api/gemini/generate',
    ...llmSecurity,
    ...llmJsonBody,
    maxPayloadSize,
    createModelAccessGuard({ allowedModels: GEMINI_ALLOWED_MODELS, defaultModel: 'gemini-1.5-flash' }),
    async (req, res) => {
    let usageEventId = null;
    try {
        const { prompt, history, files } = req.body;
        const modelName = req.allowedModel;

        const pricing = await resolvePricingForModel(req.userClient, modelName);
        const inputTokenEstimate = estimateTokensFromText(
            [...collectTextFragments(prompt), ...collectTextFragments(history)].join(' ')
        );
        const outputTokenBudget = 1024;
        const reservedCost = roundCost(
            (inputTokenEstimate / 1_000_000) * pricing.inputPricePerMillion +
            (outputTokenBudget / 1_000_000) * pricing.outputPricePerMillion
        );

        const reservation = await reserveUsageEvent(req.userClient, {
            provider: pricing.provider,
            model: modelName,
            reservedCost,
            reservedInputTokens: inputTokenEstimate,
            reservedOutputTokens: outputTokenBudget,
        });

        if (!reservation?.allowed) {
            return res.status(402).json({ error: 'Usage limit exceeded', reason: reservation?.reason ?? 'LIMIT' });
        }

        usageEventId = reservation?.usage_event_id;

        const response = await genAI.models.generateContent({
            model: modelName || "gemini-1.5-flash",
            contents: [
                ...(history || []),
                { role: 'user', parts: [{ text: prompt }] }
            ]
        });

        if (usageEventId) {
            try {
                const usageMeta = response?.usageMetadata;
                const metaUsage = usageMeta
                    ? {
                        inputTokens: toTokenCount(usageMeta.promptTokenCount),
                        outputTokens: toTokenCount(usageMeta.candidatesTokenCount),
                    }
                    : null;

                const inputCost = roundCost((toTokenCount(metaUsage?.inputTokens) / 1_000_000) * pricing.inputPricePerMillion);
                const outputCost = roundCost((toTokenCount(metaUsage?.outputTokens) / 1_000_000) * pricing.outputPricePerMillion);
                const totalCost = roundCost(inputCost + outputCost);

                await finalizeUsageEvent(req.userClient, {
                    usageEventId,
                    cost: metaUsage ? totalCost : reservedCost,
                    inputTokens: metaUsage?.inputTokens ?? inputTokenEstimate,
                    outputTokens: metaUsage?.outputTokens ?? outputTokenBudget,
                });
            } catch (finalizeError) {
                console.error('[USAGE] Failed to finalize usage_event:', finalizeError.message);
            }
        }

        res.json({ text: response.text });
    } catch (error) {
        if (usageEventId) {
            try {
                await cancelUsageEvent(req.userClient, { usageEventId, errorMessage: error.message });
            } catch (cancelError) {
                console.error('[USAGE] Failed to cancel usage_event:', cancelError.message);
            }
        }
        console.error('Gemini Error:', error.message);
        res.status(500).json({ error: 'Failed to fetch from Gemini' });
    }
});

// --- Routes ---

app.get('/api/db/conversations', authenticateUser, async (req, res) => {
    console.log(`[DB] Fetching conversations for: ${req.user.email}`);
    try {
        const { data, error } = await req.userClient
            .from('conversations')
            .select('*')
            .order('updated_at', { ascending: false });
        if (error) throw error;
        console.log(`[DB] Success: found ${data.length} chats`);
        res.json(data);
    } catch (error) {
        console.error('[DB] Error:', error.message);
        res.status(500).json({ error: error.message });
    }
});

/**
 * Database Proxy: Fetch Conversation Detail
 */
app.get('/api/db/conversations/:id', authenticateUser, async (req, res) => {
    try {
        const { id } = req.params;
        const { data: nodes, error: nErr } = await req.userClient
            .from('nodes')
            .select('*')
            .eq('conversations_id', id);
        if (nErr) throw nErr;

        const { data: messages, error: mErr } = await req.userClient
            .from('messages')
            .select('*')
            .in('nodes_id', nodes.map(n => n.id))
            .order('ordinal', { ascending: true });
        if (mErr) throw mErr;

        res.json({ nodes, messages });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * Database Proxy: Create Conversation
 */
app.post('/api/db/conversations', ...dbWriteSecurity, ...smallJsonBody, async (req, res) => {
    try {
        const payload = { ...req.body, user_id: req.user.id };
        const { data, error } = await req.userClient.from('conversations').insert(payload).select().single();
        if (error) throw error;
        res.json(data);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * Database Proxy: Create Node
 */
app.post('/api/db/nodes', ...dbWriteSecurity, ...smallJsonBody, async (req, res) => {
    try {
        const payload = { ...req.body, user_id: req.user.id };
        const { data, error } = await req.userClient.from('nodes').insert(payload).select().single();
        if (error) throw error;
        res.json(data);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * Database Proxy: Update Node Title
 */
app.patch('/api/db/nodes/:id', ...dbWriteSecurity, ...smallJsonBody, async (req, res) => {
    try {
        const { id } = req.params;
        const { title } = req.body;
        const { error } = await req.userClient
            .from('nodes')
            .update({ title })
            .eq('id', id);
        if (error) throw error;
        res.json({ success: true });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * Database Proxy: Create Message
 */
app.post('/api/db/messages', ...dbWriteSecurity, ...messageJsonBody, async (req, res) => {
    try {
        const payload = { ...req.body, user_id: req.user.id };
        const { data, error } = await req.userClient.from('messages').insert(payload).select().single();
        if (error) throw error;
        res.json(data);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * Database Proxy: Save Completed Turn and Track Usage
 */
app.post(
    '/api/db/messages/complete-turn',
    ...dbWriteSecurity,
    ...messageJsonBody,
    createModelAccessGuard({ allowedModels: TRACKED_ALLOWED_MODELS }),
    async (req, res) => {
    try {
        const {
            nodes_id,
            input_tokens = 0,
            output_tokens = 0,
            user_message,
            model_message,
        } = req.body;
        const model = req.allowedModel;

        if (
            !nodes_id ||
            !model ||
            !user_message ||
            typeof user_message.content !== 'string' ||
            typeof user_message.ordinal !== 'number' ||
            !model_message ||
            typeof model_message.content !== 'string' ||
            typeof model_message.ordinal !== 'number'
        ) {
            return res.status(400).json({ error: 'Missing required completed-turn fields.' });
        }

        const inputTokens = toTokenCount(input_tokens);
        const outputTokens = toTokenCount(output_tokens);
        const pricing = await resolvePricingForModel(req.userClient, model);
        const inputCost = roundCost((inputTokens / 1_000_000) * pricing.inputPricePerMillion);
        const outputCost = roundCost((outputTokens / 1_000_000) * pricing.outputPricePerMillion);
        const totalCost = roundCost(inputCost + outputCost);

        const messagePayloads = [
            {
                nodes_id,
                user_id: req.user.id,
                role: 'user',
                content: user_message.content,
                ordinal: user_message.ordinal,
                i_o_tokens: inputTokens,
                cost: inputCost,
                model,
                provider: pricing.provider,
            },
            {
                nodes_id,
                user_id: req.user.id,
                role: 'model',
                content: model_message.content,
                ordinal: model_message.ordinal,
                i_o_tokens: outputTokens,
                cost: outputCost,
                model,
                provider: pricing.provider,
            }
        ];

        const { data: insertedMessages, error: insertError } = await req.userClient
            .from('messages')
            .insert(messagePayloads)
            .select('*');

        if (insertError) {
            throw insertError;
        }

        let usageTracked = true;
        let caps = null;

        try {
            await recordUsageAggregate(req.userClient, {
                userId: req.user.id,
                model,
                provider: pricing.provider,
                inputTokens,
                outputTokens,
                totalCost,
            });

            await recordUserAggregate(req.userClient, req.user, totalCost);
        } catch (usageError) {
            usageTracked = false;
            console.error('[USAGE] Failed to update aggregates:', usageError.message);
        }

        try {
            const { data: capRow, error: capError } = await req.userClient.rpc('usage_get_status').single();
            if (capError) {
                throw capError;
            }
            caps = capRow;
        } catch (capErr) {
            console.error('[USAGE] Failed to load cap status:', capErr.message);
        }

        res.json({
            messages: insertedMessages,
            usage: {
                provider: pricing.provider,
                model,
                input_tokens: inputTokens,
                output_tokens: outputTokens,
                input_cost: inputCost,
                output_cost: outputCost,
                total_cost: totalCost,
                usage_tracked: usageTracked,
            },
            caps,
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * Database Proxy: Update Conversation State
 */
app.patch('/api/db/conversations/:id', ...dbWriteSecurity, ...smallJsonBody, async (req, res) => {
    try {
        const { id } = req.params;
        const { error } = await req.userClient
            .from('conversations')
            .update({ ...req.body, updated_at: new Date().toISOString() })
            .eq('id', id);
        if (error) throw error;
        res.json({ success: true });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * Database Proxy: Delete Conversation
 */
app.delete('/api/db/conversations/:id', ...dbWriteSecurity, async (req, res) => {
    try {
        const { id } = req.params;
        const { error } = await req.userClient
            .from('conversations')
            .delete()
            .eq('id', id);
        if (error) throw error;
        res.json({ success: true });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * Database Proxy: Report Bug
 */
app.post('/api/db/bugs', ...dbWriteSecurity, ...bugReportJsonBody, async (req, res) => {
    try {
        const payload = { ...req.body, user_id: req.user.id };
        const { data, error } = await req.userClient.from('bug_reports').insert(payload).select().single();
        if (error) throw error;
        res.json(data);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

/**
 * Supabase Config Proxy (Public ANON Key Only)
 * Used by the frontend ONLY for managing user sessions (Auth).
 * Database CRUD is handled by /api/db routes.
 */
app.get('/api/config/supabase', (req, res) => {
    res.json({
        url: process.env.SUPABASE_URL,
        key: process.env.SUPABASE_KEY // This should be the ANON key
    });
});

app.use((err, req, res, next) => {
    if (err?.type === 'entity.too.large') {
        return res.status(413).json({ error: 'Payload too large' });
    }

    if (err instanceof SyntaxError && 'body' in err) {
        return res.status(400).json({ error: 'Invalid JSON payload' });
    }

    next(err);
});

// Health check
app.get('/health', (req, res) => {
    res.status(200).send('OK');
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
