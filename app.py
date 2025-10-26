# app.py (FastAPI)
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from predict import predict_tweet

#criando app
app = FastAPI(title="API de Classificação de Tweets")


#criando endpoint para a API
@app.post("/predict")
async def predict(text: str = Form(...)):
    """Endpoint para classificar um tweet"""
    result = predict_tweet(text)
    return result


@app.get("/", response_class=HTMLResponse)
async def home():
    """Página inicial simples com um formulário HTML"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Classificador de Tweets</title>
        <style>
            body { font-family: Arial; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { margin-top: 20px; }
            textarea { width: 100%; height: 100px; margin-bottom: 10px; }
            button { padding: 10px 20px; background-color: #4CAF50; color: white; border: none; cursor: pointer; }
            .result { margin-top: 20px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>Classificador de Tweets de Desastre</h1>
        <p>Digite um tweet para verificar se está relacionado a um desastre:</p>

        <div class="container">
            <form id="predict-form">
                <textarea id="tweet-text" placeholder="Digite o texto do tweet aqui..."></textarea>
                <button type="submit">Classificar</button>
            </form>

            <div class="result" id="result" style="display: none;"></div>
        </div>

        <script>
            document.getElementById('predict-form').addEventListener('submit', async (event) => {
                event.preventDefault();
                const text = document.getElementById('tweet-text').value;

                if (!text) {
                    alert('Por favor, digite um tweet');
                    return;
                }

                // Enviar para a API
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: `text=${encodeURIComponent(text)}`
                });

                const result = await response.json();

                // Mostrar resultado
                const resultElement = document.getElementById('result');
                resultElement.style.display = 'block';
                resultElement.innerHTML = `
                    <h3>Resultado:</h3>
                    <p><strong>Classificação:</strong> ${result.class}</p>
                    <p><strong>Probabilidade:</strong> ${(result.probability * 100).toFixed(2)}%</p>
                    <details>
                        <summary>Ver detalhes</summary>
                        <p><strong>Texto processado:</strong> ${result.processed_text}</p>
                    </details>
                `;
            });
        </script>
    </body>
    </html>
    """
    return html_content


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)