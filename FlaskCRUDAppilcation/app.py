from flask import Flask, render_template, request, redirect
import pymysql

app = Flask(__name__)

# Database connection function
def get_connection():
    return pymysql.connect(
        host="localhost",
        user="root",
        password="root",
        database="flask_crud"
    )

# --- READ (Show all records) ---
@app.route('/')
def index():
    conn = get_connection() 
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students")
    data = cursor.fetchall()
    conn.close()
    return render_template("index.html", students=data)


# --- CREATE (Add record) ---
@app.route('/add', methods=['GET', 'POST'])
def add():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        age = request.form['age']

        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO students(name, email, age) VALUES (%s, %s, %s)",
            (name, email, age)
        )
        conn.commit()
        conn.close()
        return redirect('/')

    return render_template("add.html")


# --- UPDATE (Edit record) ---
@app.route('/edit/<int:id>', methods=['GET', 'POST'])
def edit(id):
    conn = get_connection()
    cursor = conn.cursor()

    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        age = request.form['age']

        cursor.execute("""
            UPDATE students 
            SET name=%s, email=%s, age=%s 
            WHERE id=%s
        """, (name, email, age, id))

        conn.commit()
        conn.close()
        return redirect('/')

    cursor.execute("SELECT * FROM students WHERE id=%s", (id,))
    student = cursor.fetchone()
    conn.close()
    return render_template("edit.html", student=student)


# --- DELETE (Remove record) ---
@app.route('/delete/<int:id>')
def delete(id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM students WHERE id=%s", (id,))
    conn.commit()
    conn.close()
    return redirect('/')


if __name__ == '__main__':
    app.run(debug=True)
