import sys

def main():
    total = 0
    for idx, line in enumerate(sys.stdin):
        line = line.strip()
        print(idx + 1, line)
        if line.endswith("ms"):
            total += float(line[:-2])
    print(f"{total:,.2f} milliseconds = {total / 1e3:,.3f} seconds = {total / 60e3:,.3f} minutes = {total / 3600e3:,.3f} hours")

if __name__ == "__main__":
    main()

# grep -Eo "[0-9]+(\.[0-9]+)? ms" stats-ceb_5x4096_min1_count.log | python read_timing.py