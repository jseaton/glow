from flask import Flask, render_template, request, url_for, flash, redirect
import subprocess
import re
import tempfile

app = Flask(__name__)

@app.route("/", methods=('GET', 'POST'))
def root():
    page = """<form action='/' method='post'>Filename:<input type='text' value='changeme.glsl' name='filename' /><textarea name='data' cols='80' rows='40'>vec4 mainImage() {
    return vec4(sin(iTime), 0., 0., 1.);
}</textarea><input type='submit' /></form>"""
    if request.method == 'POST':
        filename = request.form['filename']
        m = re.match(r'^([a-z0-9-]+)\.glsl$', filename)
        assert m
        head = m.groups()[0]

        glsl  = f"shaders/{head}.glsl"
        temp  = f"temp/{head}.spv"
        final = f"compiled/{head}.spv"

        with open(glsl, "w") as f:
            f.write(request.form['data'])

        page = "<p>submitted, result:</p>" + page

        state = subprocess.run(f"cat prefix.glsl {glsl} postfix.glsl | glslangValidator -V --stdin -S frag -o {temp}", shell=True, capture_output=True, text=True)

        if state.returncode == 0:
            subprocess.run(["mv", temp, final])
            page = "<p>Success!</p>" + page
        else:
            page = "Failure (" + str(state.returncode) + "):" + state.stdout + state.stderr + page
        
    return "<h1>Shader Dome</h1>" + page
