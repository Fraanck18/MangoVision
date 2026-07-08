# Levanta el tunel ngrok con el dominio fijo. Ejecutar en su PROPIA terminal
# (deja esta ventana abierta; si la cierras, el link se cae con ERR_NGROK_3200).
$ngrok = "C:\Users\ander\AppData\Local\Microsoft\WinGet\Packages\Ngrok.Ngrok_Microsoft.Winget.Source_8wekyb3d8bbwe\ngrok.exe"
& $ngrok http --url=competent-poisonously-tena.ngrok-free.dev 8000
