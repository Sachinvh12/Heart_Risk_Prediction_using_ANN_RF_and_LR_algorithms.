def funcy():
	import cgi
	form = cgi.FieldStorage()
	searchterm =  form.getvalue('Age')
	print(searchterm)
	return render_template('edaran.html')