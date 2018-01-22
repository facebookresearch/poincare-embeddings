PYTHON=python3

mammal_closure.tsv: noun_closure.tsv
	cat noun_closure.tsv | grep -e "\smammal.n.01" | cut -f1 | sed 's/\(.*\)/\^\1/g' > mammals.txt
	cat noun_closure.tsv | grep -f "mammals.txt" | grep -v -f mammals_filter.txt > mammal_closure.tsv

noun_closure.tsv:
	$(PYTHON) transitive_closure.py

clean:
	rm mammal_closure.tsv
	rm noun_closure.tsv
	rm mammals.txt
