# InstanceSpace-streamlit
A streamlit app to visualise the ISA results

To run the app, you need to have the ISA results in the data folder

```bash
streamlit run app/Homepage.py
```

To run with authentication, you need to have the credentials in the `.streamlit/secrets.toml` file

```toml
[credentials]
usernames = { example = { email = "example@example.com", name = "Example", password = "EXAMPLE_PASSWORD" }, admin = { email = "admin", name = "Admin", password = "ADMIN_PASSWORD" } }

[cookie]
expiry_days = 1
key = "some_signature_key"
name = "some_cookie_name"

[preauthorized]
emails = ["example@example.com"]

```
