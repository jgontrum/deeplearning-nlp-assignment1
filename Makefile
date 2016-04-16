
venv: bin/activate
bin/activate: requirements.txt
	test -d bin || virtualenv -p python3 .
	virtualenv --relocatable .
	( \
		source bin/activate; \
		pip install --upgrade pip; \
		pip install -Ur requirements.txt; \
	)
	touch bin/activate

clean:
	rm -rf bin include lib pip-selfcheck.json
