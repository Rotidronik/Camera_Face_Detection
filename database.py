import sqlite3
import pickle

import sqlite3
import pickle


class FaceDatabase:
    def __init__(self, db_name="faces.db"):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS users 
                              (id INTEGER PRIMARY KEY, name TEXT, encoding BLOB)''')
        self.conn.commit()

    def save_face(self, name, embedding):
        # Az embedding (numpy array) átalakítása byte-okká
        data = pickle.dumps(embedding)
        self.cursor.execute(
            "INSERT INTO users (name, encoding) VALUES (?, ?)", (name, data))
        self.conn.commit()

    def get_all_faces(self):
        self.cursor.execute("SELECT name, encoding FROM users")
        rows = self.cursor.fetchall()
        # Visszaalakítjuk a byte-okat vissza tömbökké
        return [(name, pickle.loads(encoding)) for name, encoding in rows]
