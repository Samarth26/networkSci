# networkSci

## Installation

To activate virtual environment and install dependencies (for windows)

```
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

## Running

## make sure virtual environment is activated

**First Time Running**  
The -s flag must be given to scrape data from dblp and store in network.json file  
The first argument is the path to the xls dataset file

```
python project.py DataScientists.xls -s -k <kmax value>
```

---

To run without scraping dblp data (must have been scraped before to produce a network.json file for the data in the xls file)

```
python project.py DataScientists.xls -k <kmax value>
```
