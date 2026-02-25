import networkx as nx
import csv
import copy

# main file for my goodreads network


def data_report(csv_read):
    print("length of csv file", len(list(csv_read)))


def open_data(filename):
    with open(filename, newline='') as csv_file:
        csv_read = csv.reader(csv_file, delimiter=',', quotechar='"')

        for row in csv_read:
            print(row)


        csv_file.seek(0)
        reader = csv.reader(csv_file)

        data_report(csv_read)


                                
    

open_data('../data/goodreads_books_dataset.csv')
    


