import argparse
from pathlib import Path

def parse_pairs(text: str):
    lines = text.splitlines()
    pairs = []
    current_role = None
    current_lines = []

    def flush():
        nonlocal current_role, current_lines
        if current_role and current_lines:
            pairs.append((current_role, "\n".join(current_lines).strip()))
        current_role = None
        current_lines = []

    for line in lines:
        if line.startswith("USER: "):
            flush()
            current_role = "USER"
            current_lines = [line[len("USER: "):]]
        elif line.startswith("ASSISTANT: "):
            flush()
            current_role = "ASSISTANT"
            current_lines = [line[len("ASSISTANT: "):]]
        else:
            if current_role is not None:
                current_lines.append(line)

    flush()
    return pairs

def group_pairs(pairs, pair_window: int):
    groups = []
    current = []
    user_count = 0

    i = 0
    while i < len(pairs):
        role, text = pairs[i]
        current.append((role, text))
        if role == "USER":
            user_count += 1

        if user_count >= pair_window:
            groups.append(current)
            current = []
            user_count = 0

        i += 1

    if current:
        groups.append(current)

    return groups

def render_group(group):
    out = []
    for role, text in group:
        out.append(f"{role}: {text}")
        out.append("")
    return "\n".join(out).strip() + "\n"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--pair-window", type=int, default=3)
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    text = input_path.read_text(encoding="utf-8", errors="ignore")
    pairs = parse_pairs(text)
    groups = group_pairs(pairs, args.pair_window)

    stem = input_path.stem
    for idx, group in enumerate(groups, 1):
        outpath = outdir / f"{stem}.chunk{idx:02d}.txt"
        outpath.write_text(render_group(group), encoding="utf-8")

    print(f"input_pairs: {len(pairs)}")
    print(f"chunk_files: {len(groups)}")
    for idx, group in enumerate(groups, 1):
        print(f"{idx:02d}: {sum(1 for r, _ in group if r == 'USER')} user turns")

if __name__ == "__main__":
    main()
