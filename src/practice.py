import webbrowser

f = open('helloworld.html','w')
name = "Akash Bhaiya"
name = name + '.jpg'
print(name)
message = """<html>
<head></head>
<body><p>Hello World!</p>
<img src = "/home/sethiamayank14/PycharmProjects/project2/src/images/"""+name+""""/>
</body>
</html>"""
print(message)
f.write(message)
f.close()

#Change path to reflect file location
filename = 'helloworld.html'
webbrowser.open_new_tab(filename)