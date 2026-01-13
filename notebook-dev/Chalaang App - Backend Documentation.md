# Chalaang App - Backend Documentation

## Overview
This project is a **Next.js** application leveraging AWS Serverless technologies. The backend logic is primarily handled via **Next.js API Routes** (`src/pages/api`) which interact with **DynamoDB** for data persistence, **Cognito** for authentication, and **S3** for file storage.

> [!IMPORTANT]
> The AI Generation feature in the Editor is currently **MOCKED** on the frontend and does not connect to a live backend AI service.

## Architecture Stack

| Component | Technology | Description |
|-----------|------------|-------------|
| **Framework** | Next.js (Pages Router) | Handles both frontend serving and backend API routes. |
| **Database** | AWS DynamoDB | NoSQL database for flexible data storage. |
| **Auth** | AWS Cognito | User identity and access management. |
| **Storage** | AWS S3 | Object storage for creatives and project files. |
| **Runtime** | Node.js | Environment for API routes and scripts. |

---

## Core Services (`src/lib`)

The backend logic is modularized into service classes located in `src/lib`.

### 1. DynamoDB Service (`src/lib/dynamodb.js`)
This is the central entry point for database interactions. It initializes the `DynamoDBDocumentClient` and exports specific service classes:
*   **UserService**: Handles user creation, retrieval, and mapping to Cognito groups.
*   **BrandService**: Manages brand profiles, including hierarchical logic (syncing brands from Client Owners to Client Managers).
*   **ProjectService**: Manages projects, including custom logic for "Retainer" project naming.
*   **AgencyService**: Manages agency entities.

### 2. Config Service (`src/lib/configService.js`)
A singleton service that manages dynamic system configurations with in-memory caching (TTL 5 minutes).
*   **Features/Settings**: Project Types, Roles, Permissions, Workflow States.
*   **File Constraints**: Defines max file sizes and allowed extensions per context (e.g., 'presentation' vs 'creative').
*   **UI Styles**: Returns Tailwind classes for dynamic styling.

### 3. S3 Service (`src/lib/s3Service.js`)
Manages file operations with AWS S3.
*   **Uploads**: Validates files (size/type) before uploading. Generates unique keys nested by project ID.
*   **Download**: Generates **Presigned URLs** for secure, temporary file access.
*   **Validation**: Enforces file type and size limits (fallback logic if DB config is missing).

### 4. Project Status Manager (`src/lib/projectStatusManager.js`)
Contains business logic to automatically transition project statuses based on the state of their creatives.
*   **Logic**:
    *   `DRAFT`: No creatives.
    *   `IN_REVIEW`: Some creatives uploaded.
    *   `COMPLETED`: All creatives approved.

---

## API Routes (`src/pages/api`)

The API follows a RESTful structure layered over the services.

### Authentication & Users
*   **`POST /api/cognito/create-user`**:
    *   Uses `CognitoIdentityProviderClient`.
    *   Performs `AdminCreateUserCommand` (suppressing emails).
    *   Sets permanent password immediately.
    *   Adds user to specified Cognito Group (Admin, AgencyAdmin, ClientOwner, etc.).
    *   Triggers a sync to DynamoDB via internal API call.
*   **`GET/POST /api/users`**: CRUD operations for the `User` table. Supports lookup by email or agency.

### Projects & Creatives
*   **`GET/POST /api/projects`**:
    *   Fetches projects based on `brandId`, `userId`, or `all` flag.
    *   Implements access control logic (e.g., Client Managers see their Owner's projects).
*   **`GET /api/creatives/[creativeId]`**: Fetches creative details.
*   **`POST /api/creatives`**: (Implicit in `create-october-creatives.mjs`) used for creating new creative entries.

### AI Generation (Current State)
*   **Frontend**: `src/components/editor/AIEditor.js` contains the UI for prompt input.
*   **Backend**: 
    *   Currently **NOT IMPLEMENTED**.
    *   The frontend uses a mock function (`handleGenerate` in `src/pages/editor/[id].js`) that simulates a 2-second delay and returns a placeholder image from Unsplash.
    *   No Gemini, Flux, or OpenAI integration exists in the active codebase.

---

## Data Flow

1.  **User Request** \-\> **Next.js Page**
2.  **Next.js Page** calls **API Route** (`/api/...`)
3.  **API Route** calls **Service** (`src/lib/...`)
4.  **Service**:
    *   Validates data (using Validators in `src/lib/validation`).
    *   Interacts with **AWS SDK** (DynamoDB/S3/Cognito).
    *   Returns data to API.

## Background Scripts
The root directory contains maintenance scripts (e.g., `create-october-creatives.mjs`) that directly use the AWS SDK to seed data or perform cleanup tasks. These bypass the API layer and interact directly with DynamoDB.
