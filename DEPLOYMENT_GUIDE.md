# Deploying Brand Brain to Vercel

## 1. Prerequisites

- A GitHub account.
- A Vercel account (linked to your GitHub).
- The `deployment/vercel` branch pushed to your GitHub repository.

## 2. Push Your Branch

First, ensure this branch is up on GitHub:

```bash
git push -u origin deployment/vercel
```

## 3. Import Project in Vercel

1. Go to your [Vercel Dashboard](https://vercel.com/dashboard).
2. Click **"Add New..."** -> **"Project"**.
3. Find your `brand-brain` repository and click **"Import"**.

## 4. Configure Project

In the "Configure Project" screen:

### Framework Preset

- Vercel should auto-detect "Other". If not, select **"Other"**.

### Root Directory

- Leave as `./` (default).

### Environment Variables

You MUST add your environment variables here. Copy them from your local `.env`.
Key variables likely include:

- `GOOGLE_API_KEY`
- `PINECONE_API_KEY`
- `POSTGRES_DB`
- `POSTGRES_USER`
- `POSTGRES_PASSWORD`
- `POSTGRES_HOST`
- `POSTGRES_PORT`

> [!IMPORTANT]
> Ensure your `POSTGRES_HOST` is accessible from the internet (e.g., Supabase, Neon, or a cloud RDS with public access), as Vercel is a cloud environment.

## 5. Deploy

1. Click **"Deploy"**.
2. Wait for the build to complete.
3. Once finished, you will get a domain (e.g., `brand-brain.vercel.app`).

## 6. Verify

Visit your new URL + `/health`:
`https://<your-app>.vercel.app/health`

It should return: `{"status": "ok"}`
