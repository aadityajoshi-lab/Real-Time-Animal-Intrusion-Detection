from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0002_userprofile_sns_subscription_arn_and_more"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="userprofile",
            name="sns_subscription_arn",
        ),
        migrations.AddField(
            model_name="userprofile",
            name="ntfy_subscribed",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="userprofile",
            name="telegram_chat_id",
            field=models.CharField(blank=True, default="", max_length=64),
        ),
    ]
