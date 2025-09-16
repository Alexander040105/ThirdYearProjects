import hashlib

def main():
    file_path = input('Enter file path to generate MD5 hash: ')
    with open(f'{file_path}', 'r') as file:
        content = file.read()
    md5_hash = hashlib.md5(content.encode()).hexdigest()
    print(f'\nThe original text: {content}')
    print(f'\nMD5 Hash: {md5_hash}')

if __name__ == '__main__':
    main()