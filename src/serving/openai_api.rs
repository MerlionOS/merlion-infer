/// OpenAI-compatible API endpoints.
/// POST /v1/chat/completions
/// GET  /v1/models
/// GET  /health
/// GET  /metrics

use crate::serving::http::{Request, Response, Method};
use alloc::format;

/// Route incoming HTTP requests to handlers.
pub fn handle_request(req: &Request) -> Response {
    match (&req.method, req.path.as_str()) {
        (Method::Get, "/health") => handle_health(),
        (Method::Get, "/v1/models") => handle_models(),
        (Method::Get, "/metrics") => handle_metrics(),
        (Method::Post, "/v1/chat/completions") => handle_chat_completions(req),
        (Method::Post, "/v1/completions") => handle_completions(req),
        (Method::Get, "/") => handle_root(),
        _ => Response::json(404, r#"{"error":"not found"}"#),
    }
}

fn handle_root() -> Response {
    Response::json(200, r#"{"name":"MerlionOS Inference","version":"0.1.0","status":"ready"}"#)
}

fn handle_health() -> Response {
    let uptime = crate::arch::x86_64::timer::uptime_secs();
    Response::json(200, &format!(
        r#"{{"status":"healthy","uptime_seconds":{}}}"#,
        uptime
    ))
}

fn handle_models() -> Response {
    Response::json(200, r#"{"object":"list","data":[{"id":"not-loaded","object":"model","owned_by":"merlionos"}]}"#)
}

fn handle_metrics() -> Response {
    let uptime = crate::arch::x86_64::timer::uptime_secs();
    let heap_used = crate::memory::heap::used();
    let phys_alloc = crate::memory::phys::allocated_bytes();

    let body = format!(
        "# HELP merlionos_uptime_seconds System uptime\nmerlionos_uptime_seconds {}\n\
         # HELP merlionos_heap_used_bytes Heap memory used\nmerlionos_heap_used_bytes {}\n\
         # HELP merlionos_phys_allocated_bytes Physical memory allocated\nmerlionos_phys_allocated_bytes {}\n",
        uptime, heap_used, phys_alloc,
    );
    Response::text(200, &body)
}

fn handle_chat_completions(req: &Request) -> Response {
    // Parse the request body (simplified JSON extraction)
    let body_str = core::str::from_utf8(&req.body).unwrap_or("");

    // Extract the last message content (very basic JSON parsing)
    let prompt = extract_last_message(body_str).unwrap_or("Hello");

    // For now, return a placeholder response since model isn't loaded
    Response::json(200, &format!(
        r#"{{"id":"chatcmpl-merlion","object":"chat.completion","choices":[{{"index":0,"message":{{"role":"assistant","content":"[MerlionOS Inference] Model not loaded. Received: {}"}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}}}"#,
        escape_json(prompt)
    ))
}

fn handle_completions(req: &Request) -> Response {
    let body_str = core::str::from_utf8(&req.body).unwrap_or("");
    let prompt = extract_field(body_str, "prompt").unwrap_or("Hello");

    Response::json(200, &format!(
        r#"{{"id":"cmpl-merlion","object":"text_completion","choices":[{{"text":"[Model not loaded] Echo: {}","index":0,"finish_reason":"stop"}}],"usage":{{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}}}"#,
        escape_json(prompt)
    ))
}

/// Very basic JSON field extraction (no full parser needed in no_std).
fn extract_field<'a>(json: &'a str, field: &str) -> Option<&'a str> {
    let key = format!(r#""{}""#, field);
    let pos = json.find(&key)?;
    let after_key = &json[pos + key.len()..];
    // Skip ": or :"
    let colon = after_key.find(':')?;
    let after_colon = after_key[colon + 1..].trim_start();
    if after_colon.starts_with('"') {
        let start = 1;
        let end = after_colon[1..].find('"')?;
        Some(&after_colon[start..start + end])
    } else {
        None
    }
}

/// Extract the content of the last message in a chat completions request.
fn extract_last_message(json: &str) -> Option<&str> {
    // Find the last "content" field
    let mut last = None;
    let mut search = json;
    while let Some(pos) = search.find(r#""content""#) {
        let after = &search[pos + 9..];
        if let Some(colon) = after.find(':') {
            let after_colon = after[colon + 1..].trim_start();
            if after_colon.starts_with('"') {
                if let Some(end) = after_colon[1..].find('"') {
                    last = Some(&after_colon[1..1 + end]);
                }
            }
        }
        search = &search[pos + 9..];
    }
    last
}

fn escape_json(s: &str) -> alloc::string::String {
    let mut out = alloc::string::String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str(r#"\""#),
            '\\' => out.push_str(r#"\\"#),
            '\n' => out.push_str(r#"\n"#),
            '\r' => out.push_str(r#"\r"#),
            _ => out.push(c),
        }
    }
    out
}
