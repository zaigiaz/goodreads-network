import networkx as nx
import csv


with open("../data/goodreads_books_dataset.csv") as csv_file:
    csv_read = csv.reader(csv_file, delimiter=',')
    print("length of csv file", len(list(csv_read)))
    

