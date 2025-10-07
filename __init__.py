
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.magic import register_cell_magic, Magics, cell_magic, magics, magics_class
from IPython.display import display
from transformers import AutoModel, AutoTokenizer
import torch
import re
import json

# Mock database connection and cursor for demonstration purposes
class MockCursor:
    def __init__(self):
        # Mock data for demonstration. Added a mock vector column.
        self._mock_data = [
            (1, 'Goroka Airport', 'Goroka', 'Papua New Guinea', 'GKA', 'AYGA', -6.081689834590001, 145.391998291, 5282, 10.0, 'N', 'Pacific/Port_Moresby', 'airport', 'OurAirports', [0.1, 0.2, 0.3]),
            (2, 'Madang Airport', 'Madang', 'Papua New Guinea', 'MAG', 'AYMD', -5.207079863548279, 145.78800082206726, 20, 10.0, 'N', 'Pacific/Port_Moresby', 'airport', 'OurAirports', [0.4, 0.5, 0.6]),
            (3, 'Mount Hagen Kagamuga Airport', 'Mount Hagen', 'Papua New Guinea', 'HGU', 'AYMH', -5.826789855957031, 144.29600524902344, 5388, 10.0, 'N', 'Pacific/Port_Moresby', 'airport', 'OurAirports', [0.7, 0.8, 0.9])
        ]
        self._description = [
            ('airport_id',), ('name',), ('city',), ('country',), ('iata',), ('icao',),
            ('latitude',), ('longitude',), ('altitude',), ('timezone',), ('dst',),
            ('tz_database_time_zone',), ('type',), ('source',), ('embedding_vector',) # Renamed mock vector column to match RAG
        ]
        self._index = 0

    def execute(self, query, params=None):
        # Simulate query execution and filter mock data if query looks like a vector search
        print(f"Mock executing query: {query}")
        if params:
            print(f"With parameters: {params}")

        # Simple mock of vector search: if query contains '<=> ?' and has params,
        # return a subset of data. In a real scenario, this would involve distance calculation.
        if '<=> ?' in query and params and len(params) > 0:
             # In a real mock, you'd calculate similarity. Here, just return the first few.
             self._current_results = self._mock_data[:min(len(self._mock_data), 3)] # Return top 3 mock results
             print(f"Mock vector search executed, returning {len(self._current_results)} results.")
        else:
             # Default mock behavior for non-vector queries
             self._current_results = self._mock_data
             print(f"Mock standard query executed, returning {len(self._current_results)} results.")


    def fetchall(self):
        print("Mock fetching all results.")
        # Return the results from the last execute call
        return self._current_results


    def description(self):
        print("Mock getting description.")
        # Returning mock description for demonstration.
        return self._description

    def close(self):
        print("Mock cursor closed.")
        pass

class MockConnection:
    def cursor(self):
        print("Mock connection creating cursor.")
        return MockCursor()

    def close(self):
        print("Mock connection closed.")
        pass

# Use a flag to indicate if the DB connection is mocked
MOCKED_DB_CONNECTION = True

# Load Hugging Face model and tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    print("Hugging Face model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading Hugging Face model: {e}")
    tokenizer = None
    model = None

def get_embedding(text):
    """Generates embedding for a given text using the loaded HF model."""
    if model is None or tokenizer is None:
        print("Hugging Face model not loaded. Cannot generate embedding.")
        return None
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        # Mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        return embeddings
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

# Define retrieve_database_entries function
def retrieve_database_entries(query_string: str, num_entries: int = 5) -> pd.DataFrame:
    """
    Retrieves relevant database entries using vector search based on a query string.

    Args:
        query_string: The natural language query string.
        num_entries: The number of top similar entries to retrieve.

    Returns:
        A Pandas DataFrame containing the retrieved entries, or an empty DataFrame
        if retrieval fails or no entries is found.
    """
    print(f"Attempting to retrieve {num_entries} database entries for query: '{query_string}'")
    if model is None or tokenizer is None:
        print("Hugging Face model not loaded. Cannot generate embedding for retrieval.")
        return pd.DataFrame()

    embedding = get_embedding(query_string)
    if embedding is None:
        print("Failed to generate embedding for the query string.")
        return pd.DataFrame()

    # Construct the SQL query for vector similarity search
    # We assume a vector column named 'embedding_vector' exists in the 'airports' table
    # and is indexed for vector search.
    # The query selects all columns and orders by the cosine distance (<=)> operator).
    # Note: In the mock, the actual ORDER BY and LIMIT won't be applied,
    # but the query structure is correct for a real MariaDB connection.
    sql_query = f"SELECT * FROM airports ORDER BY embedding_vector <=> ? LIMIT {num_entries};"
    print(f"Generated SQL query for vector search: {sql_query}")

    conn = None
    try:
        if MOCKED_DB_CONNECTION:
            print("Using mocked database connection for retrieval.")
            conn = MockConnection() # Use the mock connection
        else:
            # In a real scenario, establish the actual connection
            # conn = mariadb.connect(user="openflights_user",
            #                        password="openflights_password",
            #                        database="openflights",
            #                        unix_socket="/tmp/mysqld.sock")
             raise ConnectionError("Database connection failed for retrieval. Using mocked connection instead.")

        cursor = conn.cursor()

        # MariaDB Vector expects a JSON array string for vector parameters
        query_params = [json.dumps(embedding)] # Pass embedding as a JSON string parameter
        cursor.execute(sql_query, tuple(query_params))

        results = cursor.fetchall()

        if not results:
            print("Vector search query executed successfully, but returned no results.")
            return pd.DataFrame()

        # Get column names from cursor description
        column_names = [desc[0] for desc in cursor.description()]

        # Convert results to Pandas DataFrame
        df = pd.DataFrame(results, columns=column_names)

        print(f"Successfully retrieved {len(df)} entries.")
        return df

    except ConnectionError as e:
        print(f"Database Connection Error during retrieval: {e}")
        print("Please ensure the MariaDB server is running and accessible.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred during database retrieval: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()
            print("Retrieval connection closed.")

# Define generate_ai_response function
def generate_ai_response(query_string: str, retrieved_data: pd.DataFrame) -> str:
    """
    Generates an AI response based on the query and retrieved database entries.

    Args:
        query_string: The natural language query string.
        retrieved_data: A Pandas DataFrame containing the retrieved database entries.

    Returns:
        A natural language response generated by the AI model.
    """
    print("Generating AI response...")
    if model is None or tokenizer is None:
        return "AI model not loaded. Cannot generate response."

    if retrieved_data.empty:
        return f"No relevant information found in the database for the query: '{query_string}'."

    # Format the retrieved data into a context string for the AI model
    # Exclude potential vector columns from the context
    data_for_context = retrieved_data.drop(columns=[col for col in retrieved_data.columns if 'vector' in col.lower()], errors='ignore')

    context_string = "Database Entries:\n"
    for index, row in data_for_context.iterrows():
        context_string += f"- {', '.join([f'{col}: {row[col]}' for col in data_for_context.columns])}\n"

    prompt = f"Based on the following database entries, answer the question: '{query_string}'\n\n{context_string}\nAnswer:"
    print(f"Generated prompt for AI model:\n{prompt[:500]}...") # Print truncated prompt

    try:
        # This is a simplified example. A real RAG would use the context
        # to guide the response generation more effectively, potentially
        # using a different model or a more complex prompting strategy.
        # For this example, we'll just acknowledge the prompt and context.

        # Simulate AI generating a response based on the prompt and context
        # A real implementation would use the loaded HF model for text generation
        # if it's capable, or a different LLM API.
        # The current sentence-transformers model is for embeddings, not text generation.
        # To demonstrate the RAG flow, we'll provide a placeholder response.

        # You would typically pass the prompt to a text generation model here.
        # Example (requires a text generation model like GPT-2, T5, etc.):
        # from transformers import pipeline
        # generator = pipeline("text-generation", model="gpt2")
        # ai_response = generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']

        # Placeholder response simulating the RAG outcome
        ai_response = f"Based on the retrieved data, here is information related to your query '{query_string}':\n{context_string}"

        print("AI response generated (placeholder).")
        return ai_response

    except Exception as e:
        print(f"An error occurred during AI response generation: {e}")
        import traceback
        traceback.print_exc()
        return "An error occurred while generating the AI response."


@magics_class
class MariaDBMagic(Magics):

    @cell_magic
    def mariadb(self, line, cell):
        """
        Executes SQL queries against a MariaDB database and displays results
        as a Pandas DataFrame with automatic visualizations for numeric columns.

        Supports vector search queries by including 'VECTOR SEARCH:' at the
        beginning of the cell followed by the query text and SQL query.

        Supports RAG pipeline execution by including 'RAG:' at the beginning
        of the cell followed by the natural language query.

        Example for standard query:
        %%mariadb
        SELECT * FROM airports LIMIT 5;

        Example for vector search:
        %%mariadb
        VECTOR SEARCH: Airports in tropical areas
        SELECT name, city, country, embedding_vector <=> ? AS similarity FROM airports ORDER BY similarity LIMIT 5;

        Example for RAG:
        %%mariadb
        RAG: Tell me about airports in Papua New Guinea.
        """
        cell = cell.strip()
        rag_query_text = None
        sql_query = cell # Default to treating the entire cell as SQL

        # Check for RAG syntax
        rag_match = re.match(r'RAG:\s*(.*)', cell, re.IGNORECASE | re.DOTALL)

        if rag_match:
            rag_query_text = rag_match.group(1).strip()
            print(f"Detected RAG request. Query text: '{rag_query_text}'")

            # Execute RAG pipeline
            retrieved_data = retrieve_database_entries(rag_query_text)
            ai_response = generate_ai_response(rag_query_text, retrieved_data)

            print("\n--- AI Response ---")
            print(ai_response)
            print("-------------------")

            # Optionally, display the retrieved data used for RAG
            if not retrieved_data.empty:
                 print("\n--- Retrieved Data (used for RAG context) ---")
                 display(retrieved_data)
                 print("---------------------------------------------")

            return # Stop processing after RAG execution

        # If not a RAG request, proceed with standard or vector search logic
        vector_search_query_text = None

        # Check for vector search syntax (re-using logic from previous step)
        vector_search_match = re.match(r'VECTOR SEARCH:\s*(.*)', cell, re.IGNORECASE | re.DOTALL)

        if vector_search_match:
            # Extract the query text for embedding and the actual SQL query
            query_parts = cell.split('\n', 1)
            if len(query_parts) > 1:
                vector_search_query_text = vector_search_match.group(1).strip()
                sql_query = query_parts[1].strip()
                print(f"Detected Vector Search. Query text for embedding: '{vector_search_query_text}'")
                print(f"Actual SQL query: '{sql_query}'")
            else:
                print("Vector search syntax detected, but no SQL query found after the query text.")
                return

        conn = None
        try:
            if MOCKED_DB_CONNECTION:
                print("Using mocked database connection for SQL execution.")
                conn = MockConnection()
            else:
                 raise ConnectionError("Database connection failed for SQL execution. Using mocked connection instead.")

            cursor = conn.cursor()

            query_params = None
            if vector_search_query_text:
                embedding = get_embedding(vector_search_query_text)
                if embedding is not None:
                    # MariaDB Vector expects a JSON array string for vector parameters
                    query_params = [json.dumps(embedding)] # Pass embedding as a JSON string parameter
                    print("Generated embedding for vector search.")
                else:
                    print("Could not generate embedding. Skipping vector search.")
                    return # Or handle error appropriately

            # Execute the query. Pass parameters if doing vector search.
            if query_params:
                 cursor.execute(sql_query, tuple(query_params))
            else:
                cursor.execute(sql_query)

            results = cursor.fetchall()

            if not results:
                print("Query executed successfully, but returned no results.")
                return

            # Get column names from cursor description
            column_names = [desc[0] for desc in cursor.description()]

            # Convert results to Pandas DataFrame
            df = pd.DataFrame(results, columns=column_names)

            # Handle display of vector columns - exclude from default display or summarize
            vector_cols_to_hide = [col for col in column_names if 'vector' in col.lower()] # Simple heuristic

            if vector_cols_to_hide:
                print(f"\nNote: Vector columns {vector_cols_to_hide} are present and may be large. Displaying a subset of columns.")
                # Create a DataFrame for display excluding vector columns, or handle as needed
                display_df = df.drop(columns=vector_cols_to_hide, errors='ignore')
            else:
                display_df = df

            print("Query results as DataFrame:")
            display(display_df)

            # Automatic visualization for numeric columns (excluding potential similarity score if added)
            numeric_cols = display_df.select_dtypes(include=['number']).columns
            # Exclude 'similarity' column if it was added for vector search display
            numeric_cols = [col for col in numeric_cols if col.lower() != 'similarity']


            if numeric_cols:
                print("\nGenerating visualizations for numeric columns:")
                num_plots = len(numeric_cols)
                # Adjust figure size based on number of plots
                fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(8, num_plots * 4))

                if num_plots == 1:
                    axes = [axes] # Ensure axes is an iterable even for a single plot

                for i, col in enumerate(numeric_cols):
                    sns.histplot(data=display_df, x=col, ax=axes[i], kde=True)
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')

                plt.tight_layout()
                plt.show()
            else:
                print("\nNo numeric columns found for visualization (excluding similarity score).")


        except ConnectionError as e:
            print(f"Database Connection Error: {e}")
            print("Please ensure the MariaDB server is running and accessible.")
        except Exception as e:
            print(f"An error occurred during SQL execution or data processing: {e}")
            # Print traceback for better debugging
            import traceback
            traceback.print_exc()
        finally:
            if conn:
                conn.close()
                print("SQL execution connection closed.")

# Register the magic automatically when the module is imported
try:
    get_ipython().register_magics(MariaDBMagic)
    print("%%mariadb cell magic registered from mariadb_colab_magic module.")
except NameError:
    print("IPython environment not found, skipping magic registration.")

