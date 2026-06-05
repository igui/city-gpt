import sys
from city_gpt_utils import generate_names, build_prompt

def main():
    # US Populated Places
    prompt = build_prompt("US", "P", "PPL")
    print(f"--- Generating 3 US Populated Places ---")
    print(f"Prompt string: {prompt}")
    
    try:
        names = generate_names(prompt_str=prompt, num_names=3, max_new_tokens=60)
        for i, name in enumerate(names, 1):
            print(f"{i}. {name}")
    except Exception as e:
        print(f"Error during generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
