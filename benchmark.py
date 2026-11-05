"""Порівняння трьох алгоритмів пошуку підрядка на двох статтях."""

from functools import partial
from pathlib import Path
from statistics import median
import timeit


def boyer_moore(text: str, pattern: str) -> int:
    """Повертає індекс першого збігу або -1."""
    size = len(pattern)
    if not size:
        return 0
    if size > len(text):
        return -1

    last_position = {char: index for index, char in enumerate(pattern)}

    # Для кожного суфікса обчислюємо зсув і межу шаблону.
    suffix_shift = [0] * (size + 1)
    border = [0] * (size + 1)
    left, right = size, size + 1
    border[left] = right
    while left > 0:
        while right <= size and pattern[left - 1] != pattern[right - 1]:
            if suffix_shift[right] == 0:
                suffix_shift[right] = right - left
            right = border[right]
        left -= 1
        right -= 1
        border[left] = right

    # Для решти суфіксів використовуємо найдовший відповідний префікс.
    right = border[0]
    for index in range(size + 1):
        if suffix_shift[index] == 0:
            suffix_shift[index] = right
        if index == right:
            right = border[right]

    last_start = len(text) - size
    start = 0
    while start <= last_start:
        index = size - 1
        while index >= 0 and pattern[index] == text[start + index]:
            index -= 1
        if index < 0:
            return start

        bad_char = text[start + index]
        bad_char_shift = index - last_position.get(bad_char, -1)
        start += max(bad_char_shift, suffix_shift[index + 1])
    return -1


def knuth_morris_pratt(text: str, pattern: str) -> int:
    """Повертає індекс першого збігу або -1."""
    size = len(pattern)
    if not size:
        return 0
    if size > len(text):
        return -1

    # prefix[i] — довжина найдовшого власного префікса pattern[:i+1],
    # який одночасно є його суфіксом.
    prefix = [0] * size
    matched = 0
    for index in range(1, size):
        while matched > 0 and pattern[index] != pattern[matched]:
            matched = prefix[matched - 1]
        if pattern[index] == pattern[matched]:
            matched += 1
        prefix[index] = matched

    matched = 0
    for index, char in enumerate(text):
        while matched > 0 and char != pattern[matched]:
            matched = prefix[matched - 1]
        if char == pattern[matched]:
            matched += 1
            if matched == size:
                return index - size + 1
    return -1


def rabin_karp(text: str, pattern: str) -> int:
    """Повертає індекс першого збігу або -1."""
    size = len(pattern)
    if not size:
        return 0
    if size > len(text):
        return -1

    base = 0x110000  # На одиницю більше за найбільший код Unicode.
    modulus = 1_000_000_007
    highest_power = pow(base, size - 1, modulus)
    pattern_hash = window_hash = 0
    for index, char in enumerate(pattern):
        pattern_hash = (pattern_hash * base + ord(char)) % modulus
        window_hash = (window_hash * base + ord(text[index])) % modulus

    if window_hash == pattern_hash and text[:size] == pattern:
        return 0

    for end in range(size, len(text)):
        outgoing = ord(text[end - size])
        incoming = ord(text[end])
        remaining_hash = window_hash - outgoing * highest_power
        window_hash = (remaining_hash * base + incoming) % modulus
        # Перевірка символів виключає хибний збіг через колізію хешів.
        if window_hash == pattern_hash:
            start = end - size + 1
            if text[start:end + 1] == pattern:
                return start
    return -1


ALGORITHMS = {
    "Боєр–Мур": boyer_moore,
    "Кнут–Морріс–Пратт": knuth_morris_pratt,
    "Рабін–Карп": rabin_karp,
}
ARTICLES = {
    "article1.txt": "ефективніше застосовувати",
    "article2.txt": "дискретні структури даних",
}
MISSING_PATTERN = "пошук у бібліотеці Марса!"
REPEATS = 7
CALLS_PER_REPEAT = 100


def main() -> None:
    data_dir = Path(__file__).resolve().parent / "data"
    totals = dict.fromkeys(ALGORITHMS, 0.0)
    print("Час одного виклику, мкс (медіана серій)\n")
    print("| Файл | Підрядок | Індекс | " + " | ".join(ALGORITHMS) + " |")
    print("|---|---|---:|---:|---:|---:|")

    for filename, existing_pattern in ARTICLES.items():
        text = (data_dir / filename).read_text(encoding="utf-8-sig")
        article_totals = dict.fromkeys(ALGORITHMS, 0.0)

        for case, pattern, must_exist in (
            ("Наявний", existing_pattern, True),
            ("Вигаданий", MISSING_PATTERN, False),
        ):
            expected = text.find(pattern)
            if (expected >= 0) != must_exist:
                raise ValueError(f"Неочікуваний результат для {pattern!r} у {filename}")

            timings = []
            for name, search in ALGORITHMS.items():
                if search(text, pattern) != expected:
                    raise ValueError(f"{name} повернув хибний індекс у {filename}")

                samples = timeit.repeat(
                    partial(search, text, pattern), repeat=REPEATS, number=CALLS_PER_REPEAT
                )
                microseconds = median(samples) / CALLS_PER_REPEAT * 1_000_000
                timings.append(f"{microseconds:.2f}")
                article_totals[name] += microseconds
                totals[name] += microseconds

            print(f"| {filename} | {case} | {expected} | " + " | ".join(timings) + " |")

        fastest = min(article_totals, key=article_totals.__getitem__)
        means = " | ".join(f"{total / 2:.2f}" for total in article_totals.values())
        print(f"| {filename} | Середнє ({fastest}) | — | {means} |")

    fastest = min(totals, key=totals.__getitem__)
    case_count = 2 * len(ARTICLES)
    means = " | ".join(f"{total / case_count:.2f}" for total in totals.values())
    print(f"| Обидва файли | Середнє ({fastest}) | — | {means} |")


if __name__ == "__main__":
    main()
