def funcy():
	import cgi
	from flask import Flask,render_template
	form = cgi.FieldStorage()
	searchterm =  form.getvalue('Age')
	print(searchterm)
	return render_template('edaran.html')