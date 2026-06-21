import sqlite3
import os

# Veritabanı dosyasının yolu
db_path = os.path.join(os.getcwd(), 'data', 'sentiment_db.sqlite')

# Bağlantı kur
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Tablodaki tüm verileri sil
cursor.execute('DELETE FROM predictions')

# ID sayacını sıfırla (Bir sonraki kayıt 1'den başlasın)
cursor.execute('DELETE FROM sqlite_sequence WHERE name="predictions"')

conn.commit()
conn.close()

print("Veritabanı başarıyla temizlendi!")