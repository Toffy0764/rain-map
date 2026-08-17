# Mappa Piogge - Ultimi 15 giorni

App Streamlit standalone che mostra la pioggia cumulata degli ultimi N giorni
(fino a 15) come mappa raster con gradiente di blu (più scuro = più pioggia),
sovrapposta a una mappa interattiva Folium.

## Come funziona

1. Costruisce una griglia di punti (lat/lon) sull'area scelta.
2. Interroga Open-Meteo (`past_days`) per ogni punto in parallelo,
   sommando la pioggia giornaliera degli ultimi N giorni completi.
3. Interpola i valori discreti su una griglia fine con `scipy.griddata`
   (metodo cubico, con fallback "nearest" per i buchi).
4. Converte la matrice interpolata in un'immagine PNG con colormap "Blues"
   e la sovrappone alla mappa come `ImageOverlay`.

## Avvio locale

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy su Streamlit Community Cloud

Stesso procedimento usato per SPX Volatility Signals e Indice Fungaiolo:
push su un repo GitHub (es. `rain-map`), poi collega il repo da
share.streamlit.io puntando a `app.py`.

## Note

- L'area predefinita copre Trentino e Veneto; è selezionabile anche
  "Personalizzata" inserendo un bounding box lat/lon a mano.
- La densità della griglia di rilevamento (numero di chiamate API) è
  separata dalla risoluzione del raster finale (qualità dell'immagine),
  per non dover moltiplicare le chiamate a Open-Meteo solo per avere
  un'immagine più definita.
- Con griglie molto fitte (>250 punti) il caricamento può richiedere
  qualche minuto: i risultati vengono comunque messi in cache per un'ora
  (`st.cache_data(ttl=3600)`).
- Se vuoi, si può in seguito riusare la stessa logica di caching/parallel
  fetch già presente nell'Indice Fungaiolo per ridurre ulteriormente i
  tempi di attesa.
