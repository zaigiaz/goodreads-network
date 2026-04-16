import csv
import ast


def parse_genres(genre_str):
    try:
        return ast.literal_eval(genre_str)
    except:
        return []


def load_goodreads_data(filepath):
    books = {}
    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            book_id = row.get('URL', '').split('/')[-1] if row.get('URL') else str(idx)
            genres = parse_genres(row.get('Genres', '[]'))
            
            rating_str = row.get('Num_Ratings', '0').replace(',', '')
            num_ratings = int(rating_str) if rating_str.isdigit() else 0
            
            books[book_id] = {
                'title': row.get('Book', ''),
                'author': row.get('Author', ''),
                'description': row.get('Description', ''),
                'genres': genres,
                'avg_rating': float(row.get('Avg_Rating', 0)),
                'num_ratings': num_ratings,
            }
    return books


if __name__ == '__main__':
    data = load_goodreads_data('../data/goodreads_data.csv')
    print(f"Loaded {len(data)} books")