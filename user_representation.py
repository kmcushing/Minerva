# could initially just be a json with most common topics (and some measure of their frequency) along with a list of domains they have insider info on
import jsons

# TODO: define an enum for topics based on how we are extracting them

# TOOO: maybe use a hierarchical structure for domains with insider knowledge?


class User:
    # TODO: may want to also store user sentiment with topics
    # TODO: better representation for user domains
    # TODO: use some sort of decay for user's frequent topics with a decay rate for each message?
    def __init__(self, id, insider_domains=None, topic_frequencies=None):
        self.id = id
        self.insider_domains = insider_domains if insider_domains else {}
        self.topic_frequencies = topic_frequencies if topic_frequencies else {}
        self.save()

    def disscussed_topic(self, topic):
        # adds to discussion count of topic in user representation
        if topic in self.topic_frequencies:
            self.topic_frequencies[topic] += 1
        else:
            self.topic_frequencies[topic] = 1
        self.save()

    def add_insider_domain(self, domain):
        # adds insider domain for user
        self.insider_domains[domain] = 1
        self.save()

    def remove_insider_domain(self, domain):
        self.insider_domains.pop(domain, None)
        self.save()

    def __str__(self):
        return f"Insider Domains: {self.insider_domains}\nTopic Frequencies: {self.topic_frequencies}"

    def to_prompt(self):
        # returns text representation of user for llm prompt
        s = "Assume the user does not have insider knowledge for all domains unless otherwise specified. "
        if self.topic_frequencies:
            s += "The user has previously discuessed the following topics:\n"
            for t in self.topic_frequencies:
                s += f"{t}: Frequency {self.topic_frequencies[t]}\n"
        if self.insider_domains:
            s += "The user has insider knowledge of the following domains:\n"
            for d in self.insider_domains:
                s += f"{d}\n"
        return s

    def save(self):
        f = open(f"data/user_representations/{self.id}.json", "w")
        f.write(jsons.dumps(self))


# load user representation
def load_user(id):
    # TODO: may want to remove hard-coded path
    try:
        f = open(f"data/user_representations/{id}.json", "r")
        return jsons.loads(f.read(), User)
    except:
        return User(id)


# user1 = load_user("user1")
# print(user1.to_prompt())
