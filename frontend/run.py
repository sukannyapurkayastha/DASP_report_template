# File for debugging with PyCharm, you don't need to run this but please don't delete it

try:
    from streamlit.web import bootstrap
except ImportError:
    from streamlit import bootstrap

real_script = 'app.py'
bootstrap.run(real_script, False, [f'run.py {real_script}'], {})