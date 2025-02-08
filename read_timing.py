import sys

def main():
    total = 0
    for idx, line in enumerate(sys.stdin):
        line = line.strip()
        print(idx + 1, line)
        if line.endswith("ms"):
            total += float(line[:-2])
    print(f"{total:,.2f} milliseconds = {total / 1000:,.2f} seconds = {total / 60000:,.2f} minutes = {total / 3600000:,.2f} hours")

if __name__ == "__main__":
    main()

# grep -Eo "[0-9]+(\.[0-9]+)? ms" stats-ceb_5x4096_min1_count.log | python read_timing.py