package maps.manuallycollected;

/**
* Form https://github.com/sherxon/AlgoDS/
*/


public class SieveofEratosthenes {


    // print all prime numbers up to given number n
    public static boolean[] findPrimes(int n) {
        boolean[] primes = new boolean[n];

        for (int i = 2; i < n; i++)
            if (!primes[i])
                for (int j = i * i; j < primes.length; j += i)
                    primes[j] = true;

        return primes;

    }
}
