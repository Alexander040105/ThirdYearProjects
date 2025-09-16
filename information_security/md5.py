import hashlib

def main():
        text = input("Enter text to generate MD5 hash: ")
        if not text.strip():
                print("No text provided. Please try again.")
                return
        md5_hash = hashlib.md5(text.encode()).hexdigest()
        print(f'\nMD5 Hash: {md5_hash}')

if __name__ == '__main__':
        main()
