#!/usr/bin/env python3
"""判断年份是闰年还是平年。

规则（公历）：
    1. 能被 4 整除但不能被 100 整除 -> 闰年；
    2. 能被 400 整除               -> 闰年；
    3. 其余情况为平年。

用法：
    python leap_year.py          # 交互式输入
    python leap_year.py 2024     # 命令行参数
"""

import sys


def is_leap_year(year: int) -> bool:
    """判断给定年份是否为闰年。"""
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def describe(year: int) -> str:
    """返回一句中文结论。"""
    if is_leap_year(year):
        return f"{year} 年是闰年（共 366 天，2 月有 29 天）。"
    return f"{year} 年是平年（共 365 天，2 月有 28 天）。"


def ask_year() -> int:
    """循环读取用户输入，直到得到一个合法的年份。"""
    while True:
        raw = input("请输入一个年份（如 2024）：").strip()
        if not raw:
            print("输入不能为空，请重新输入。")
            continue
        try:
            year = int(raw)
        except ValueError:
            print(f"“{raw}” 不是合法的整数年份，请重新输入。")
            continue
        if year < 1:
            print("年份应为正整数，请重新输入。")
            continue
        return year


def pause_if_interactive(no_pause: bool = False) -> None:
    """在交互式控制台里暂停，避免双击运行时窗口瞬间关闭。"""
    if no_pause:
        return
    try:
        if not sys.stdin or not sys.stdin.isatty():
            return  # 被重定向或由其他程序调用，不停顿
    except Exception:
        return
    try:
        input("\n按回车键退出...")
    except (EOFError, KeyboardInterrupt):
        pass


def main() -> int:
    no_pause = "--no-pause" in sys.argv[1:]
    args = [a for a in sys.argv[1:] if a != "--no-pause"]

    try:
        if args:
            try:
                year = int(args[0])
            except ValueError:
                print(f"参数 “{args[0]}” 不是合法年份。")
                return 1
        else:
            year = ask_year()

        print(describe(year))
        return 0
    finally:
        pause_if_interactive(no_pause)


if __name__ == "__main__":
    sys.exit(main())
