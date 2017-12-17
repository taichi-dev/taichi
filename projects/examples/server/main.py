import taichi as tc

app = tc.get_pakua_server()

app.run(port=1111 , host='0.0.0.0')
