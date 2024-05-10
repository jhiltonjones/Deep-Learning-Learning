import re

def extract_data(log_file_path):
    steps = []
    normalized_means = []
    pattern = re.compile(r"Normalized LOG: steps (\d+), episodes.*returns ([\d.-]+)")
    pattern2 = re.compile(r":Normalized LOG: steps (\d+), episodes.*returns ([\d.-]+)")

    with open(log_file_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            notmatch = pattern2.search(line)
            if match and not notmatch:
                step = int(match.group(1))
                normalized_mean = float(match.group(2))
                steps.append(step)
                normalized_means.append(normalized_mean)
    return normalized_means

# Usage
def main():
    log_file_path = 'Walker2d/Walker2d/expert/0/4_run/log'
    normalized_means = extract_data(log_file_path)
    # print("mean_returns = ", normalized_means)

if __name__  == "__main__":
    main()