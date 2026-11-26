import { appendFileSync } from "node:fs";
import { randomUUID } from "node:crypto";

// Requests are serial within a Pi agent. A retry supersedes an unfinished request;
// the unmatched request remains in the audit and never becomes an SFT example.
export default function (pi) {
  let requestId = null;
  const record = (value) => appendFileSync("/logs/agent/sft-capture.jsonl", JSON.stringify(value) + "\n");
  pi.on("before_provider_request", (event) => {
    const p = event.payload;
    if (!Array.isArray(p?.messages)) throw new Error("Missing provider messages");
    const q = {...p, max_tokens:16384, temperature:0.7, top_p:0.8, top_k:20,
      chat_template_kwargs:{...(p.chat_template_kwargs||{}),enable_thinking:false}};
    delete q.reasoning_effort;
    delete q.max_completion_tokens;
    requestId = randomUUID();
    const settings = {model:q.model,max_tokens:q.max_tokens,temperature:q.temperature,
      top_p:q.top_p,top_k:q.top_k,enable_thinking:false,at:Date.now()};
    appendFileSync("/logs/agent/request-settings.jsonl", JSON.stringify(settings)+"\n");
    record({type:"request",id:requestId,at:Date.now(),messages:q.messages,tools:q.tools||[],settings});
    return q;
  });
  pi.on("message_end", (event) => {
    if (event.message?.role !== "assistant") return;
    record({type:"response",id:requestId,at:Date.now(),message:event.message});
    requestId = null;
  });
}
