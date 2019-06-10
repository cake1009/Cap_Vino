from requests import get  # to make GET request

def download(url, file_name):
    with open(file_name, "wb") as file:   # open in binary mode
        response = get(url)               # get request
        file.write(response.content)      # write to file

if __name__ == '__main__':
	url = "https://firebasestorage.googleapis.com/v0/b/cap-vino.appspot.com/o/images%2F87da01bd8047adb0.jpg?alt=media&token=36461d10-f88d-4337-97e7-57227739a344"
	download(url,"iml.jpg")
    