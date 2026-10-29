/** Trusted native-tool bridge. No model-controlled executable, shell, or filesystem tool. */
import {readFileSync} from 'node:fs';
import {spawn} from 'node:child_process';

export default function (pi) {
  const pin = JSON.parse(readFileSync(process.env.FLOCK_TOOLS_PIN, 'utf8'));
  const allowed = pin.toolset.tools.map(t => t.name);
  pi.on('before_provider_headers', event => {
    event.headers['X-Session-ID'] = process.env.FLOCK_PI_SESSION_ID;
  });
  for (const tool of pin.toolset.tools) {
    pi.registerTool({
      name: tool.name, label: tool.name, description: tool.description,
      parameters: tool.parameters,
      async execute(_id, args, signal) {
        return await new Promise((resolve, reject) => {
          const child = spawn(process.env.FLOCK_PI_PYTHON,
            ['-m', 'flock.tools.cli', tool.name, '--json', '-'],
            {shell: false, stdio: ['pipe', 'pipe', 'pipe'], env: process.env});
          let output = '';
          // Drain diagnostics without reflecting worker details to the model.
          const abort = () => child.kill('SIGTERM');
          signal?.addEventListener('abort', abort, {once: true});
          if (signal?.aborted) abort();
          child.stdout.on('data', chunk => { output += chunk; });
          child.stderr.on('data', () => {});
          child.on('error', reject);
          child.on('close', code => {
            signal?.removeEventListener('abort', abort);
            const text = output || (code ? 'Flock tool failed; inspect worker logs.' : '{}');
            resolve({content: [{type: 'text', text}], details: {error: code !== 0}});
          });
          child.stdin.on('error', () => {});
          child.stdin.end(JSON.stringify(args));
        });
      }
    });
  }
  pi.on('session_start', () => pi.setActiveTools(allowed));
}
