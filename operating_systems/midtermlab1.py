file_extensions = ['txt','log','csv','html','xml','json','bat','exe','pdf','docx']
import os

current_dir = os.getcwd()
if os.path.exists('CMD_Projects'): 
    os.chdir(f'{current_dir}/CMD_Projects')    
else:
    os.mkdir('CMD_Projects')
    os.chdir(f'{current_dir}/CMD_Projects')

current_dir = os.getcwd()
print(f'The current directory is: {current_dir}')
counter = 1
for extensions in file_extensions:
    for i in range(len(file_extensions)):
        print(f'testFile{counter}.{extensions}')
        counter += 1
        try:
            with open(f'testFile{counter}.{extensions}', 'x') as file:
                file.write(f"This is the test file {counter} of type {extensions}")
                print(f"This is the test file {counter} of type {extensions}")
        except FileExistsError:
            print("Error: File 'new_exclusive_file.txt' already exists.")