# UKAR - Uncertainty-aware knowledge acquisition 

<img width="2448" height="816" alt="UKAR" src="https://github.com/user-attachments/assets/cc30fe22-c844-4cc5-b149-1346e502cc21" />


***
**Problem.**  A Small local AI model often invents a count, date, rate, or measurement that was never in the chat. Some setups then send the whole question to another model, so you never see which fact was missing.

**Solution.**  UKAR uses the llama.cpp or Ollama model already running on your computer. That model names only the missing facts. UKAR fills the ones you supplied, or the ones in a local table you added. The same model writes the answer. If a fact is still missing, you get the question instead of a made-up number.

It does not load a model of its own, browse the web, or call another computer. A question the model already knows is answered by that same model. "Already trusted" in a trace means the model claimed the fact. It does not mean a table checked it.

## How it works

The local model makes the "can answer now" call as its gap report. UKAR then fills facts. A partial fill still asks you. The package ships with the facts you pass in. A local table is used only after you add one.

```text
Question
   |
   v
Local model
   |
   +--> Can answer now?
   |          |
   |          +--> Yes --> Same model writes the answer
   |
   +--> No
           |
           v
      Missing facts
           |
           v
          UKAR
           |
           +--> Facts you passed in
           +--> Local tables you added
           |
           v
   Every required fact filled?
           |
      +----+----+
      |         |
     No        Yes
      |         |
      v         v
  Ask user   Same model
             writes the answer
```

## What you can do with it

- Keep the answer on your PC. UKAR only talks to `127.0.0.1`, `localhost`, or `::1`. A public address is refused before any request is sent.
- UKAR is designed so acquired numeric facts come from facts you supplied or from local sources on your computer. A model can still state a number it claims to already know.
- See the gap. The trace shows how sure the model was, which facts it named, which ones a table filled, and which questions are still open.
- Know when UKAR ran. `ukar demo` and `ukar ask` add `UKAR was used for this reply.` when the trace is answered or still waiting on you. In the tool examples, that line is added only after `ukar_acquire` returns a result that is not an error. If the model never calls the tool, the line is not added.
- Bring your own rows. This package does not ship a topic, a rate card, or a worked example in one subject. Add a local table when you have facts that should be filled the same way every time.

UKAR does not pick a smarter model for hard questions. It does not browse the web. llama.cpp and Ollama also do not load this project from a plugin folder. You either run `ukar ask`, or you pass the `ukar_acquire` tool on the chat request and run it when the model calls it.

## Examples

Three runs. The first two need no model server. The third uses whatever model Ollama already has loaded. llama.cpp is the same command with `--backend llamacpp` and `--base-url http://127.0.0.1:8080`.

### No facts yet

The offline stand-in has no topic, so it stops and names the missing fact.

```bash
python -m ukar demo
```

```text
UKAR
The local model names missing facts. Sources return those facts. The same model answers.

Question
  What is still missing before you can answer?

Status  waiting_on_user
Backend scripted

Confidence before acquisition  0.2
Missing facts
  user_fact        personal   A value only the user knows — Without that value the reply would have to guess  [user_fact]

Still needed before an answer
  - A value only the user knows — reply with user_fact. Without that value the reply would have to guess

Facts still missing, so the local model was not asked to guess.
Facts and the answer stay on this machine.

UKAR was used for this reply.
```

### You already know the fact

Pass it with `--fact`. The stand-in repeats that fact and does not add another number.

```bash
python -m ukar demo --fact stool_count=4
```

```text
Status  answered
Backend scripted

Already trusted
  - stool_count = 4

Acquired
  stool_count            4        given            0¢  local
    Supplied with the question: stool_count = 4.

Local answer
  Using only the facts that were supplied:
  - stool_count = 4

UKAR was used for this reply.
```

### A question the loaded model already knows

This run used Ollama `qwen3:4b`. The model said it could answer, put `50` in its own known list, and the reply repeated that. "Already trusted" means the model claimed it knew the fact. No table on the computer checked it. Another model may word the same question differently.

```bash
python -m ukar ask "How many states in the US?" \
  --backend ollama \
  --base-url http://127.0.0.1:11434 \
  --model qwen3:4b
```

```text
Question
  How many states in the US?

Status  answered
Backend ollama

Confidence before acquisition  1.0
Already trusted
  - 50

Local answer
  50

Facts and the answer stay on this machine.

UKAR was used for this reply.
```

### A number only you know

Ask the loaded model. If the count is not something it can know, it should name that fact and leave the number out. When you know it, pass it and ask again:

```bash
python -m ukar ask "How many stools fit in the studio?" \
  --backend ollama \
  --base-url http://127.0.0.1:11434 \
  --model qwen3:4b \
  --fact stool_count=4
```

## Try it before you connect a model

Python 3.11 or newer. No extra packages.

```bash
python -m pip install -e .
python -m ukar demo
```

From a checkout, `PYTHONPATH=.` works instead of the install.

The demo uses a stand-in with no topic, so llama.cpp and Ollama can be off. With no facts, it stops and names the missing fact. It does not invent one. Pass a fact you already know and it repeats that fact and nothing else:

```bash
python -m ukar demo --fact stool_count=4
```

```bash
python -m unittest discover -s tests -v
```

## Use your llama.cpp server

Start `llama-server` on this PC with `--jinja` and a model that can call tools. Qwen 2.5 Instruct, Llama 3.1 and newer, and Mistral Nemo are the templates the server documents. Details: [llama.cpp function calling](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md).

```bash
llama-server -m model.gguf --jinja --host 127.0.0.1 --port 8080
```

Ask UKAR to run the question on that server:

```bash
python -m ukar ask "How many stools fit in the studio?" \
  --backend llamacpp \
  --base-url http://127.0.0.1:8080
```

Or hand the model the tool and let it call `ukar_acquire` during the chat:

```bash
python examples/llamacpp_plugin.py \
  "How many stools fit in the studio?" \
  --base-url http://127.0.0.1:8080
```

The example posts to `http://127.0.0.1:8080/v1/chat/completions`. When the model calls the tool, the script runs it locally and sends the result back as `role: tool` with the `tool_call_id` from llama-server. It stops after four rounds.

If the server log says `Chat format: Generic`, tool calling is less reliable. Use a GGUF with a tool template, or pass one with `--chat-template-file`. That flag is llama.cpp's.

## Use your Ollama server

Ollama accepts the same tool on `POST /api/chat`. Details: [Ollama tool calling](https://docs.ollama.com/capabilities/tool-calling). Use a model that supports tools. `qwen3:4b` is a small one. Ollama's docs use the `qwen3` name. Pass the tag you pulled.

```bash
ollama serve
ollama pull qwen3:4b
```

```bash
python -m ukar ask "How many stools fit in the studio?" \
  --backend ollama \
  --base-url http://127.0.0.1:11434 \
  --model qwen3:4b
```

Or let the model call the tool:

```bash
python examples/ollama_plugin.py \
  "How many stools fit in the studio?" \
  --model qwen3:4b
```

The gap turn asks Ollama for JSON. The answer turn is normal text. A tool result goes back as `{"role": "tool", "tool_name": "ukar_acquire", "content": "..."}`. The loop stops after four rounds.

Any question uses the same loop. If the loaded model says it already knows the answer, it answers. If the answer depends on a fact only you know, it is supposed to name that fact. It can still be wrong if the model claims it already knows the number. Pass `--fact key=value` when you already have the fact. `--model` on Ollama must be the tag already loaded. If you omit it, the command asks Ollama for `llama3.2`. llama.cpp uses the GGUF that server was started with.

## Add it to your own chat loop

```python
from ukar.plugin import SYSTEM_PROMPT, execute, tool_schemas

tools = tool_schemas()
# Pass `tools` on the llama-server or Ollama chat request.
# When the model returns a call named ukar_acquire:
text = execute("ukar_acquire", {"question": "...", "needs_json": "{...}", "facts": "stool_count=4"})
```

Put `SYSTEM_PROMPT` in the system message so the model knows to name missing facts and then call the tool. `needs_json` is the gap object as a string. `facts` is `key=value` pairs or a JSON object. `execute` fills keys from that `facts` argument. It does not open a socket, and it does not read a table file unless you change the sources yourself.

The gap object looks like this:

```json
{"confidence": 0.2, "can_answer_now": false, "known": [], "needs": [
  {"id": "user_fact", "statement": "The value only this user knows", "why": "The reply depends on that value", "scope": "personal", "freshness": "static", "keys": ["user_fact"]}
]}
```

`scope` is `personal`, `public`, or `computed`. `freshness` is `static`, `daily`, or `live`. A report that names another model (`delegate_to`, `cloud_model`, `route_to`, and the same kind of key) is refused.

## Bring your own tables

A table is a source with `cover(needs, known)`, `quote(need_ids)`, and `fetch(needs, known)`. `fetch` gets the missing facts and the values you already passed in. It does not get a way to call another model. `privacy` must be `local`. Anything else is skipped, and `fetch` is never called.

```python
from ukar import Router, default_sources
from ukar.backends import ScriptedCompleter

trace = Router(ScriptedCompleter(), sources=default_sources()).run(
    "How many stools fit in the studio?",
    {"stool_count": "4"},
)
print(trace.answer)
```

`ScriptedCompleter` is the stand-in behind `python -m ukar demo`. Use it to check your tables without starting a model server.

## License and your own risk

This project is MIT. Copyright (c) 2026 starrshaw. See `LICENSE`.

The license says the software is provided **as is**, with no warranty of any kind. The author and copyright holder are not liable for any claim, damages, or other liability from using it. You use UKAR at your own risk.

An answer is not professional advice. You are responsible for the facts you put in and for what you do with the answer.

The code in this repository is written for this project and is MIT. It has no runtime dependencies. It does not copy in llama.cpp, Ollama, or another license. Those programs stay where you installed them. llama.cpp and Ollama are MIT projects of their own. You need Python to run this package. Python itself is under the PSF License, which is not MIT, and it is not included here.

## Important

**UKAR is a software tool that reduces guessed facts. It is not a guarantee that an answer is correct, and it is not professional advice.**

**An answer may still be wrong if:**

- **a supplied fact is wrong,**
- **a local table contains incorrect information,**
- **a model incorrectly claims it already knows a fact,**
- **a model reasons incorrectly from correct facts,**
- **a needed fact is not identified.**

**Always verify outputs before using them for legal, financial, medical, engineering, safety-critical, or regulatory decisions.**
