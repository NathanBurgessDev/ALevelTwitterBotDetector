# ALevelTwitterBotDetector
A level project built to detect twitter bots

Based on a towards data science post by Daniel Faucher:
https://towardsdatascience.com/python-detecting-twitter-bots-with-graphs-and-machine-learning-41269205ab07

Model's a twitter users "user space" by scraping their tweets and replies and producing a graph based on how frequently other users interact with the "main" user and eachother within this "user space". This allows you to determine the impact a user (node) has on a user space.

An XGBoost linear regression model was trained on these user spaces to attempt to categorise the difference between a humans user space and a bots user space.

No longer works :( - Cant scrape data from twitter anymore.
