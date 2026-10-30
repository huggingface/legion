"""Explicit agent-facing schemas; lifecycle and grading are never tools."""


def schema(name, description, properties=None, required=()):
    return {'name': name, 'description': description, 'parameters': {
        'type': 'object', 'properties': properties or {}, 'required': list(required),
        'additionalProperties': False}}


STRING = {'type': 'string', 'maxLength': 65536}
NODE = {'node_id': {'type': 'string'}, 'arguments': {'type': 'object'}}
ESCAPE_TOOLS = [
    schema('get_observation', 'Read the room description; progress is shared by the whole team.'),
    schema('investigate', 'Inspect a visible entity.', {'node_id': STRING}, ['node_id']),
    schema('tool_call', 'Execute an investigated tool with its named inputs.', NODE,
           ['node_id', 'arguments']),
    schema('use_item', 'Activate an item with its named inputs.', NODE, ['node_id', 'arguments']),
    schema('submit_answer', 'Submit the victory string; only verified success ends the episode.',
           {'answer': STRING}, ['answer']),
]
TERMINAL_TOOLS = [
    schema('get_observation', 'Read task instructions and shared command status.'),
    schema('exec_start', 'Start one bounded command in the shared task container. '
           'Other writes are busy until it finishes. Do not daemonize commands.',
           {'command': STRING, 'timeout_s': {'type': 'integer', 'minimum': 1, 'maximum': 300}},
           ['command']),
    schema('exec_status', 'Check a command handle.', {'job_id': STRING}, ['job_id']),
    schema('exec_output', 'Read a bounded page of completed command output.',
           {'job_id': STRING, 'offset': {'type': 'integer', 'minimum': 0}}, ['job_id']),
    schema('exec_cancel', 'Cancel a command and end this attempt; partial state is not graded.',
           {'job_id': STRING}, ['job_id']),
    schema('fs_read', 'Read at most 32 KiB of a sandbox file, between mutations.',
           {'path': STRING, 'offset': {'type': 'integer', 'minimum': 0}}, ['path']),
    schema('fs_write', 'Write a UTF-8 sandbox file, between commands.',
           {'path': STRING, 'content': STRING}, ['path', 'content']),
    schema('task_submit', 'Finalize the shared sandbox for native grading. First submission wins.'),
]


def tools_for(benchmark):
    if benchmark == 'agent_escape':
        return ESCAPE_TOOLS
    if benchmark == 'terminal_bench':
        return TERMINAL_TOOLS
    raise ValueError(f'Unknown benchmark: {benchmark}')
