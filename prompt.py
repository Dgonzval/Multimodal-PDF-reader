prompt_image ='''


'''

prompt_final_response = '''

Consulta: {query}\n\nRespuesta del texto: {text_response}\n\nRespuesta de la imagen: {image_response}\n\nElabora una respuesta final en torno a estos elementos.

Asegurate que la respuesta sea concreta para eso sigue los siguientes pasos:

-Analiza la respuesta textual.
-Analiza la respueta de la imagen.
-Si coinciden entre ellas genera una respuesta en base a ambas respuestas.
-Si no coinciden genera una respuesta en base a la que tenga más sentido con repecto a la [query].
-Revisa tu respuesta y asegúrate que tenga sentido.
-Genera respuesta final concreta:

[Respuesta]

'''

