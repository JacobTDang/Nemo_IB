"""Phase B2: validate .mcp.json declares all 5 MCP servers.

Asserts:
- File exists at the project root and parses as JSON
- All 5 expected servers listed under mcpServers
- Each entry has command, args, and env
- args invoke `python -m <module> server` with the literal "server" arg
- The python interpreter path is resolvable from the project root
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS = os.path.join(ROOT, '.mcp.json')

EXPECTED_SERVERS = {
  'nemo_web':       'tools.web_search_server.web_search',
  'nemo_financial': 'tools.financial_modeling_engine.analysis_tools',
  'nemo_finnhub':   'tools.news_agregator.finnhub_server',
  'nemo_fred':      'tools.news_agregator.fred_server',
  'nemo_alpaca':    'tools.alpaca.server',
}


def _load():
  assert os.path.exists(SETTINGS), f"missing: {SETTINGS}"
  with open(SETTINGS, 'r', encoding='utf-8') as f:
    return json.load(f)


def test_file_exists_and_parses():
  data = _load()
  assert isinstance(data, dict), "top level must be an object"
  print("PASS: .mcp.json parses as JSON object")


def test_all_servers_registered():
  data = _load()
  mcp = data.get('mcpServers', {})
  missing = [name for name in EXPECTED_SERVERS if name not in mcp]
  assert not missing, f"missing servers: {missing}"
  extras = [name for name in mcp if name not in EXPECTED_SERVERS]
  assert not extras, f"unexpected servers (rename or drop): {extras}"
  print(f"PASS: all {len(EXPECTED_SERVERS)} servers registered")


def test_each_server_has_command_args_env():
  data = _load()
  mcp = data['mcpServers']
  for name in EXPECTED_SERVERS:
    e = mcp[name]
    assert 'command' in e, f"{name}: missing command"
    assert 'args' in e and isinstance(e['args'], list) and e['args'], \
      f"{name}: args must be a non-empty list"
    assert 'env' in e and isinstance(e['env'], dict), \
      f"{name}: env must be an object"
  print("PASS: every server has command + args + env")


def test_args_use_module_path():
  data = _load()
  mcp = data['mcpServers']
  for name, expected_module in EXPECTED_SERVERS.items():
    args = mcp[name]['args']
    assert '-m' in args, f"{name}: args must invoke python -m"
    m_idx = args.index('-m')
    actual_module = args[m_idx + 1]
    assert actual_module == expected_module, \
      f"{name}: expected module {expected_module!r}, got {actual_module!r}"
  print("PASS: every server invokes the correct python module")


def test_args_pass_server_subcommand():
  """Every server's __main__ requires the literal 'server' arg to enter
  stdio mode — without it the process prints usage and exits 0, which
  Claude Code reports as a connection failure."""
  data = _load()
  mcp = data['mcpServers']
  for name in EXPECTED_SERVERS:
    args = mcp[name]['args']
    assert 'server' in args, f"{name}: args missing 'server' subcommand"
    # 'server' must come AFTER '-m <module>', not before it
    m_idx = args.index('-m')
    srv_idx = args.index('server')
    assert srv_idx > m_idx + 1, \
      f"{name}: 'server' must follow -m <module>; got args={args}"
  print("PASS: every server gets the required 'server' subcommand")


def test_pythonpath_in_env():
  data = _load()
  mcp = data['mcpServers']
  for name in EXPECTED_SERVERS:
    pp = mcp[name]['env'].get('PYTHONPATH')
    assert pp, f"{name}: env.PYTHONPATH must be set so modules resolve"
  print("PASS: PYTHONPATH set on every server")


def test_python_interpreter_is_resolvable():
  data = _load()
  mcp = data['mcpServers']
  for name in EXPECTED_SERVERS:
    cmd = mcp[name]['command']
    if cmd.startswith('./') or cmd.startswith('.\\'):
      resolved = os.path.join(ROOT, cmd.lstrip('./').lstrip('.\\'))
      assert os.path.exists(resolved), \
        f"{name}: command {cmd!r} not resolvable from project root"
  print("PASS: every command path is resolvable")


def test_modules_import_cleanly():
  for module in EXPECTED_SERVERS.values():
    try:
      __import__(module)
    except Exception as e:
      raise AssertionError(f"{module}: import failed -> {type(e).__name__}: {e}")
  print(f"PASS: all {len(EXPECTED_SERVERS)} declared modules import cleanly")


if __name__ == "__main__":
  test_file_exists_and_parses()
  test_all_servers_registered()
  test_each_server_has_command_args_env()
  test_args_use_module_path()
  test_args_pass_server_subcommand()
  test_pythonpath_in_env()
  test_python_interpreter_is_resolvable()
  test_modules_import_cleanly()
  print("\nAll Phase B2 settings validation tests passed.")
