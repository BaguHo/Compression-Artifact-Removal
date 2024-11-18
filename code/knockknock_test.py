from knockknock import discord_sender

webhook_url = "<webhook_url_to_your_discord_channel>"


@discord_sender(webhook_url=webhook_url)
def train_your_nicest_model(your_nicest_parameters):
    import time

    time.sleep(10000)
    return {"loss": 0.9}  # Optional return value
