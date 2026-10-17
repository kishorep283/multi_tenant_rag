from rag.models import DocumentParent

def retrieve_parent_documents(child_results):
    """
    Retrieve unique parent documents from ranked child results.
    """

    # 1. Collect unique parent IDs in ranking order.
    parent_ids = list(
        dict.fromkeys(
            result["parent_id"]
            for result in child_results
            if result.get("parent_id") is not None
        )
    )

    if not parent_ids:
        return []

    # 2. Fetch all matching parents in one database query.
    parents = DocumentParent.objects.filter(
        id__in=parent_ids
    )

    # 3. Create a lookup for the retrieved parents.
    parent_lookup = {
        parent.id: parent
        for parent in parents
    }

    # 4. Preserve the order of the ranked child results.
    retrieved_parents = []

    for parent_id in parent_ids:
        parent = parent_lookup.get(parent_id)

        if parent is None:
            continue

        retrieved_parents.append({
            "id": parent.id,
            "source_filename": parent.source_filename,
            "content": parent.content,
            "metadata": parent.metadata,
        })

    return retrieved_parents
