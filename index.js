const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const { OpenRouter } = require('@openrouter/sdk');
const { OpenAI } = require('openai');
const { GoogleGenAI } = require('@google/genai');
const axios = require('axios');

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
app.use(express.json({ limit: '50mb' })); // Support base64 images

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

app.use((req, res, next) => {
    if (req.body) {
        req.body = sanitizeBody(req.body);
    }
    next();
});

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
const llmRateLimiter = rateLimit({
    windowMs: 60 * 1000, // 1 minute
    max: 20, // 20 requests per minute
    keyGenerator: (req) => req.user ? req.user.id : req.ip, // Limit per properly tracked user instead of shared IP
    message: { error: 'Too many requests, please try again later.' }
});

const maxPayloadSize = (req, res, next) => {
    // 5MB limit roughly (base64 images can be big, but prevent massive memory exhaustion)
    if (req.body && JSON.stringify(req.body).length > 5000000) {
        console.warn(`[SECURITY] Blocked oversized payload from User: ${req.user?.id || 'Unknown'}`);
        return res.status(413).json({ error: 'Payload too large' });
    }
    next();
};

const llmSecurity = [authenticateUser, llmRateLimiter, maxPayloadSize];

// --- Service Clients ---

const openRouterClient = new OpenRouter({
    apiKey: process.env.OPENROUTER_API_KEY,
});

const openaiClient = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

const genAI = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

// --- Routes ---

/**
 * OpenRouter Proxy with Streaming Support
 */
app.post('/api/openrouter/chat', llmSecurity, async (req, res) => {
    try {
        const { model, messages, stream } = req.body;

        const response = await axios.post('https://openrouter.ai/api/v1/chat/completions', {
            model,
            messages,
            stream: stream || false,
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

            response.data.pipe(res);
        } else {
            res.json(response.data);
        }
    } catch (error) {
        console.error('OpenRouter Error:', error.response?.data || error.message);
        res.status(error.response?.status || 500).json({ error: 'Failed to fetch from OpenRouter' });
    }
});

/**
 * OpenAI Proxy
 */
app.post('/api/openai/chat', llmSecurity, async (req, res) => {
    try {
        const { model, messages } = req.body;
        const completion = await openaiClient.chat.completions.create({
            model,
            messages,
        });
        res.json(completion);
    } catch (error) {
        console.error('OpenAI Error:', error.message);
        res.status(500).json({ error: 'Failed to fetch from OpenAI' });
    }
});

/**
 * Gemini Proxy
 */
app.post('/api/gemini/generate', llmSecurity, async (req, res) => {
    try {
        const { model: modelName, prompt, history, files } = req.body;
        const response = await genAI.models.generateContent({
            model: modelName || "gemini-1.5-flash",
            contents: [
                ...(history || []),
                { role: 'user', parts: [{ text: prompt }] }
            ]
        });
        res.json({ text: response.text });
    } catch (error) {
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
app.post('/api/db/conversations', authenticateUser, async (req, res) => {
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
app.post('/api/db/nodes', authenticateUser, async (req, res) => {
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
app.patch('/api/db/nodes/:id', authenticateUser, async (req, res) => {
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
app.post('/api/db/messages', authenticateUser, async (req, res) => {
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
 * Database Proxy: Update Conversation State
 */
app.patch('/api/db/conversations/:id', authenticateUser, async (req, res) => {
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
app.delete('/api/db/conversations/:id', authenticateUser, async (req, res) => {
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
app.post('/api/db/bugs', authenticateUser, async (req, res) => {
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

// Health check
app.get('/health', (req, res) => {
    res.status(200).send('OK');
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
