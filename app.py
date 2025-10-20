"""
langchainを用いたAIエージェント
"""

import re
import subprocess
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

model = ChatOllama(model="elyza:jp8b")


def call_model(prompt: str) -> str:
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)
    return response.content


# コーディングエージェント: 自然言語からPythonコードを生成
def coding_agent(spec: str) -> str:
    prompt = (
        "以下の仕様を満たすPythonコードを書いてください。"
        "コード以外の文言は絶対に回答に含めないでください。"
        f"仕様: {spec}"
    )
    return call_model(prompt)


# テストエージェント: 生成されたコードを受けてテストコードを作成
def test_code_agent(code: str, src_file: str) -> str:
    prompt = (
        "以下のPythonコードに対して、pytest形式のテストコードを作成してください。"
        "コード以外の文言は絶対に回答に含めないでください。"
        "・コードの先頭に、このコードで使われている外部モジュールをすべてimportしてください。"
        f"・テストコードの先頭に、対象関数のインポート文を入れてください。対象モジュール名は{src_file}です。"
        "・テスト関数はtest_で始めてください。"
        "・少なくとも3つのassertテストケースを作成してください。"
        f"コード:\n{code}\n"
    )
    return call_model(prompt)


# ファイル保存ヘルパー
def save_file(filename: str, content: str):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)


# pytest実行
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


# 修正依頼用コーディングエージェント呼び出し
def fix_code_agent(code: str, test_result: str) -> str:
    prompt = (
        f"以下のPythonコードがあります。\n{code}\n"
        f"このコードのテストで次のエラーが発生しました。\n{test_result}\n"
        "エラーが出ないようにコードを修正してください。修正版コードのみ返してください。"
        "コード以外の文言は絶対に回答に含めないでください。"
    )
    return call_model(prompt)


def clean_code_block(code: str) -> str:
    code = code.strip()
    # 先頭に ``` や ```
    code = re.sub(r"^```(?:python)?\n?", "", code)
    # 末尾に ```
    code = re.sub(r"\n?```$", "", code)
    return code.strip()


def is_test_file_valid(pytest_result: str) -> bool:
    error_indicators = [
        "NameError",
        "SyntaxError",
        "ImportError",
        "IndentationError",
        "ModuleNotFoundError",
        "failed collecting",
    ]

    lower_result = pytest_result.lower()
    for err in error_indicators:
        if err.lower() in lower_result:
            return False
    return True


def main():
    spec = "1+2をする簡単な処理を作成して。"
    src_file = "gen.py"
    test_file = "test_gen.py"

    # 1. コーディングエージェントがコード生成
    code = coding_agent(spec)
    print("生成されたコード:\n", code)

    # 2. テストエージェントがテストコード生成
    test_code = test_code_agent(code, src_file)
    print("生成されたテストコード:\n", test_code)

    code = clean_code_block(code)
    test_code = clean_code_block(test_code)

    # ファイル保存
    save_file(src_file, code)
    save_file(test_file, test_code)

    for i in range(5):
        print(f"=== テストサイクル {i+1} ===")
        result = run_pytest(test_file)
        print("pytest結果:\n", result)

        if not is_test_file_valid(result):
            print("テストファイルに致命的なエラーがあります。修正を依頼します。")
            code = fix_code_agent(code, result)
            code = clean_code_block(code)
            save_file(src_file, code)
            test_code = test_code_agent(code, src_file)
            test_code = clean_code_block(test_code)
            save_file(test_file, test_code)
            continue

        if "failed" not in result.lower() and "error" not in result.lower():
            print("テスト合格！ 終了します。")
            break

        print("テスト失敗、コード修正を依頼します。")
        code = fix_code_agent(code, result)
        code = clean_code_block(code)
        save_file(src_file, code)


if __name__ == "__main__":
    main()
