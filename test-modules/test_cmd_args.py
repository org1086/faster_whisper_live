from email.policy import default
import click
import sys

# @click.command()
# @click.option('--name', '-n', default='')

def getFilePath():
    file_path = None
    if len(sys.argv) == 2:
        file_path = sys.argv[1]
    return file_path

def main():
    file_path = getFilePath()
    if file_path == None:
        returnResult('')
    else:
        speech2text(file_path)


def speech2text(file_path):
    # process
    f = open(file_path, "r")
    print(f.readline())

    return returnResult(file_path)

def returnResult(result):
    print(result)

if __name__ == '__main__':
    main()