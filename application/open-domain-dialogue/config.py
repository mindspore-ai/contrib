from functions import load_csv, nug_eval

# Configurations
CSV_PATH = "testdata.csv"
W_1 = 10
W_2 = 5
W_3 = 3
K = 5
L = 3

# Main workflow
def main():
    data = load_csv(CSV_PATH)  # retrieve turn-level scores etc (EnDex framework is used)
    results = nug_eval(data, W_1, W_2, W_3, K, L)  # calculate nugget-level scores
    print(results)

if __name__ == "__main__":
    main()