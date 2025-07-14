from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    z_max = None
    punto = None
    if request.method == "POST":
        a = float(request.form["a"])
        b = float(request.form["b"])
        c = float(request.form["c"])
        x_max = float(request.form["x_max"])
        y_max = float(request.form["y_max"])

        vertices = [(0, 0), (0, min(y_max, c)), (min(x_max, c), 0)]
        if x_max + y_max > c:
            inter1 = c - x_max
            if 0 <= inter1 <= y_max:
                vertices.append((x_max, inter1))
            inter2 = c - y_max
            if 0 <= inter2 <= x_max:
                vertices.append((inter2, y_max))

        z_vals = [(a * x + b * y, (x, y)) for x, y in vertices]
        z_max, punto = max(z_vals)

    return render_template("index.html", z_max=z_max, punto=punto)

if __name__ == "__main__":
    app.run(debug=True)
