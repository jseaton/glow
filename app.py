from flask import Flask, render_template, request, url_for, flash, redirect
import subprocess
import re
import tempfile

app = Flask(__name__)

intro = "<pre>" + open("intro.txt", "r").read() + "</pre>"

@app.route("/", methods=('GET', 'POST'))
def root():
    page = ""
    shader = """// CC BY-NC-SA 3.0 Anonymous

vec4 mainImage() {
    return vec4(sin(iTime), 0., 0., 1.);
}"""
    if request.method == 'POST':
        shader = request.form['data']
        filename = request.form['filename']
        m = re.match(r'^([a-z0-9][a-z0-9-]+)\.glsl$', filename)
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

            with open("current", "w") as current:
                current.write(glsl)

            page = "<p>Success!</p>" + page
        else:
            page = f"Failure {state.returncode}: <pre>{state.stderr}</code><pre>{state.stdout}</code>{page}"
        
    form = f"<form action='/' method='post'><div><label for='filename'>Filename</label><input type='text' value='changeme.glsl' name='filename' /></div><div><textarea name='data' cols='80' rows='40'>{shader}</textarea></div><input type='submit' /></form>"

    return "<h1>Shader Dome</h1>" + page + form + intro
