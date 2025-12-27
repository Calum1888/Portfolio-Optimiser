This project combines Machine Learning with Financial Engineering. It uses several AI models to predict whether a stock will go up or down, and then calculates the "Perfect Mix" of those stocks to minimize your risk while hitting your profit goals.

How it Works
AI Predictions: The script trains four different AI models (Random Forest, XGBoost, etc.) for every stock you pick. It looks at technical indicators like RSI and ATR to decide if a stock is a "Buy," "Hold," or "Sell."

Smart Voting: It uses a "Majority Vote" system—the AI models vote on the best move to ensure the prediction is as accurate as possible.

Risk Management: Using the Ledoit-Wolf model, it calculates how stocks move together (correlation).

Optimization: Finally, it uses math to find the Optimal Weights (how much of each stock to own) to give you the lowest possible "Risk" (Volatility) for your chosen return.
