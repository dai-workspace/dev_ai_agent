"""
langchain、langgraphを用いたAIエージェント
"""

import re
import subprocess
from typing import TypedDict

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END


# ==== LLMモデル ====
model = ChatOllama(model="elyza:jp8b")


def call_model(prompt: str) -> str:
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)
    return response.content


# ==== ステート定義 ====
class State(TypedDict):
    spec: str
    src_file: str
    test_file: str
    code: str
    test_code: str
    pytest_result: str


# ==== ヘルパー ====
def clean_code_block(code: str) -> str:
    code = code.strip()
    code = re.sub(r"^```(?:python)?\n?", "", code)
    code = re.sub(r"\n?```$", "", code)
    return code.strip()


def save_file(filename: str, content: str):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)


def run_pytest(filename: str) -> str:
    try:
        result = subprocess.run(
            ["pytest", filename, "--maxfail=1", "--disable-warnings"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        return result.stdout
    except Exception as e:
        return str(e)


def is_test_file_valid(pytest_result: str) -> bool:
    error_indicators = [
        "NameError",
        "SyntaxError",
        "ImportError",
        "IndentationError",
        "ModuleNotFoundError",
        "failed collecting",
    ]
    for err in error_indicators:
        if err.lower() in pytest_result.lower():
            return False
    return True


# ==== エージェント ====
def coding_agent(state: State) -> State:
    prompt = (
        "以下の仕様を満たすPythonコードを書いてください。"
        "コード以外の文言は絶対に回答に含めないでください。"
        f"仕様: {state['spec']}"
    )
    code = clean_code_block(call_model(prompt))
    state["code"] = code
    save_file(state["src_file"], code)
    return state


def test_agent(state: State) -> State:
    prompt = (
        "以下のPythonコードに対して、pytest形式のテストコードを作成してください。"
        "コード以外の文言は絶対に回答に含めないでください。"
        "・コードの先頭に、このコードで使われている外部モジュールをすべてimportしてください。"
        f"・テストコードの先頭に、対象関数のインポート文を入れてください。対象モジュール名は{state['src_file']}です。"
        "・テスト関数はtest_で始めてください。"
        "・少なくとも3つのassertテストケースを作成してください。"
        f"コード:\n{state['code']}\n"
    )
    test_code = clean_code_block(call_model(prompt))
    state["test_code"] = test_code
    save_file(state["test_file"], test_code)
    return state


def run_test(state: State) -> State:
    result = run_pytest(state["test_file"])
    state["pytest_result"] = result
    return state


def fix_code_agent(state: State) -> State:
    prompt = (
        f"以下のPythonコードがあります。\n{state['code']}\n"
        f"テストで次のエラーが発生しました。\n{state['pytest_result']}\n"
        "エラーが出ないように修正版コードのみ返してください。"
        "コード以外の文言は絶対に回答に含めないでください。"
    )
    fixed = clean_code_block(call_model(prompt))
    state["code"] = fixed
    save_file(state["src_file"], fixed)
    return state


# ==== 遷移条件 ====
def decide_next(state: State) -> str:
    result = state["pytest_result"].lower()

    if not is_test_file_valid(state["pytest_result"]):
        return "fix"

    if "failed" not in result and "error" not in result:
        return "end"

    return "fix"


# ==== グラフ構築 ====
graph = StateGraph(State)

graph.add_node("codegen", coding_agent)
graph.add_node("testgen", test_agent)
graph.add_node("run_pytest", run_test)
graph.add_node("fix", fix_code_agent)

graph.add_edge("codegen", "testgen")
graph.add_edge("testgen", "run_pytest")
graph.add_conditional_edges("run_pytest", decide_next, {"fix": "fix", "end": END})
graph.add_edge("fix", "testgen")

graph.set_entry_point("codegen")
workflow = graph.compile()


# ==== 実行 ====
if __name__ == "__main__":
    init_state: State = {
        "spec": "1+2をする簡単な処理を作成して。",
        "src_file": "gen.py",
        "test_file": "test_gen.py",
        "code": "",
        "test_code": "",
        "pytest_result": "",
    }

    final_state = workflow.invoke(init_state, config={"recursion_limit": 30})

    print("==== 結果 ====")
    print("コード:\n", final_state["code"])
    print("テストコード:\n", final_state["test_code"])
    print("テスト結果:\n", final_state["pytest_result"])
